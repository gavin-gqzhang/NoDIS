import copy
import glob
import json
import math
import os
import os.path
import random
import re
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from maskrcnn_benchmark.modeling.roi_heads.relation_head.attention_blocks import Attn_block,Trans_block
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.comm import all_gather_with_grad, concat_all_gather, get_rank,find_linear_layers
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.make_layers import make_fc
import transformers
import logging
from .attention_blocks import MLP,fusion_func
from transformers import BertConfig,BertModel,AutoTokenizer

class NoDIS(nn.Module):
    def __init__(self, config, in_channels, statistics,baseline_model="PENet"):
        super().__init__()
        self.config=config
        self.logger=logging.getLogger(__name__)
        
        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        
        self.baseline_model=baseline_model
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.mlp_dim = in_channels
        self.k_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM         
        self.v_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM    
        
        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        assert self.num_rel_cls == len(rel_classes)
        self.rel_classes = rel_classes
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        rel_cls_weight=torch.tensor([1/per_num for per_num in per_predicate_num],dtype=torch.float)
        self.rel_cls_weight=rel_cls_weight/(rel_cls_weight.sum())
        
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        self.obj_embed = nn.Embedding(len(obj_classes), self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.filter_pred_prot=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        if baseline_model=='PENet':
            self.project_prot_head = MLP(2048*2, self.mlp_dim,self.hidden_dim,2)
        else:
            self.project_prot_head = MLP(self.mlp_dim, self.mlp_dim,self.hidden_dim,2)
        
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # contrast learning
        
        # *************************** generate predicate reps based on union and entity pair reps ***************************
        if baseline_model=='PENet':
            self.cps_t_sub_reps,self.cps_t_obj_reps=MLP(2048,self.mlp_dim//2,self.hidden_dim,1),MLP(2048,self.mlp_dim//2,self.hidden_dim,1)
        else:
            self.cps_t_sub_reps,self.cps_t_obj_reps=MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)
            
        self.cps_entity_pair_reps,self.gate_vis_entity=MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)
        
        # *************************** entity node pair --> predicate reps ***************************
        self.cps_union_reps=MLP(self.pooling_dim,self.mlp_dim//2,self.hidden_dim,1)
        self.use_node_branch=True
        if self.use_node_branch:
            self.edge_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
            self.node_to_pre=nn.ModuleList([
                nn.ModuleList([
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),  # generate predicate reps
                    nn.LayerNorm(self.hidden_dim),
                    MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),  # proj predicate reps -> predicate prototype 
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # Cross attention predicate prototype           
                ]) for _  in range(rel_layer)
            ])
            
            self.refine_edge_pre=nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        nn.LayerNorm(self.hidden_dim),
                        nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True), 
                        nn.LayerNorm(self.hidden_dim),
                        MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1), # Enhance entity weight in union features
                    ]),
                    nn.Sequential(
                        nn.Linear(2*self.hidden_dim,self.hidden_dim),
                        nn.Sigmoid()
                    ),  # del entity features
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance predicate prototye reps in union reps
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate)  # Refine predicate reps
                ]) for _ in range(rel_layer)
            ])

            self.logger.info('init node relation representation branch......')
            self.edg_branch_emp_weight,self.pos_edg_branch_scores,self.neg_edg_branch_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
            
        # *************************** union triple --> predicate reps ***************************
        self.use_denoise_branch=config.MODEL.ROI_RELATION_HEAD.USE_DENOISE_BRANCH
        if self.use_denoise_branch:
            self.noise_factor=nn.Parameter(torch.ones(1),requires_grad=True)  # add noise
            
            self.denoise_modules=nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.ReLU(inplace=True)
                    ), # denoise
                    nn.ModuleList([  # denoise subject/object features
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # subject self attention
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # object self attention
                        nn.Sequential(
                            nn.Linear(2*self.hidden_dim,self.hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(self.hidden_dim,self.hidden_dim)
                        ),
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # predicate reps attention sub-obj reps
                        nn.Sequential(
                            nn.Linear(2*self.hidden_dim,self.hidden_dim),
                            nn.Sigmoid()
                        ),  # del entity features
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # t_predicate reps attention union features
                    ])
                ]) for _ in range(rel_layer)
            ])
            
            self.denoise_reps=nn.Sequential(
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2)
                    )
            
            self.logger.info('init denoise relation representation branch......')
            self.recon_branch_emp_weight,self.pos_recon_branch_scores,self.neg_recon_branch_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
            
        # **************** Semantic consistency module ****************
        self.align_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.filter_noise_rel=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # **************** discriminator module ****************
        self.step=config.MODEL.ROI_RELATION_HEAD.TRAIN_STEP
        if self.step!=1:
            self.build_diff_modules()
                
        self.use_branch_fusion=config.MODEL.ROI_RELATION_HEAD.USE_BRANCH_FUSION
        if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
            self.filter_tri=nn.Sequential(
                nn.Sigmoid(),
                nn.Dropout(0.2)
            )
            self.filter_recon=nn.Sequential(
                nn.Sigmoid(),
                nn.Dropout(0.2)
            )   

            self.logger.info('init fusion node and denoise reps branch......')
            self.sum_branch_emp_weight,self.pos_sum_branch_scores,self.neg_sum_branch_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
        
        # **************** process global features ****************
        self.use_global_vis_refine=config.MODEL.ROI_RELATION_HEAD.USE_GLOBAL_VISUAL
        if self.use_global_vis_refine:
            self.ds_glob_reps=nn.Sequential(
                nn.Linear(5*config.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,self.hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim,self.hidden_dim)
            )
            
            self.proj_glob_reps = MLP(self.hidden_dim, self.hidden_dim, self.mlp_dim*2, 2)
            self.filter_glob_reps=nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.glob_refine_rel_reps=nn.ModuleList([
                nn.ModuleList([
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim*2,self.hidden_dim,dropout_rate) if self.use_node_branch else nn.Identity(),  # for edg rel reps
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim*2,self.hidden_dim,dropout_rate) if self.use_denoise_branch else nn.Identity(),  # for denoise rel reps
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim*2,self.hidden_dim,dropout_rate) if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch else nn.Identity(),  # for sum rel reps
                    ]) for _ in range(rel_layer)
                ])
        
        self.use_global_rel_reps=config.MODEL.ROI_RELATION_HEAD.USE_GLOBAL_REPRESENTATION
        if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
            self.global_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.mlp_dim*2,)))
            self.merge_rel_reps=nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(self.mlp_dim*2),
                    nn.MultiheadAttention(self.mlp_dim*2,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim*2),
                    nn.MultiheadAttention(self.mlp_dim*2,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim*2),
                    nn.MultiheadAttention(self.mlp_dim*2,num_head,dropout=dropout_rate,batch_first=True), 
                    nn.LayerNorm(self.mlp_dim*2),
                    MLP(self.mlp_dim*2,self.mlp_dim//2,self.mlp_dim*2,1)
                ]) for _ in range(rel_layer)
            ])
            self.glob_branch_emp_weight,self.pos_glob_branch_scores,self.neg_glob_branch_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
        
        self.use_kl_modules=config.MODEL.ROI_RELATION_HEAD.USE_KL_MODULE
        self.use_kl_weight_loss=config.MODEL.ROI_RELATION_HEAD.USE_KL_REWEIGHT_LOSS
        if self.use_kl_modules and not self.use_kl_weight_loss:
            self.kl_infos=nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.mlp_dim*2,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim,self.mlp_dim),
                    nn.LayerNorm(self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim,self.mlp_dim*2)
                ),  # mean
                nn.Sequential(
                    nn.Linear(self.mlp_dim*2,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim,self.mlp_dim),
                    nn.LayerNorm(self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim,self.mlp_dim*2)
                )  # log std
            ])
                
        # self.predict_method=config.MODEL.ROI_RELATION_HEAD.PRE_RESULT
        # assert self.predict_method=="sum" or ( self.predict_method==None and self.use_glob_refine_modules ), print(f'if predict method is not sum, please check using glob refine module')
        # ******************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999

        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
        
        self.use_adaptive_loss=config.MODEL.ROI_RELATION_HEAD.USE_ADAPTIVE_REWEIGHT_LOSS
        self.gt_scores,self.emp_decay=torch.zeros(self.num_rel_cls,requires_grad=False),0.8
    
    def build_diff_modules(self,flow_depth=14):
        self.fdm=FDM(self.config,self.mlp_dim*2,self.mlp_dim,self.num_rel_cls,pre_codebook=True)
        setattr(self.fdm.diff_module,'rel_cls_weight',self.rel_cls_weight)
        setattr(self.fdm,'num_rel_cls',self.num_rel_cls)
        setattr(self.fdm,'init_proto_loss',self.init_proto_loss)

    def freeze_module(self):
        pass
    
    def previous_res(self):
        logger=logging.getLogger(__name__)
        if self.step!=1:
            pre_step_res=torch.load(f'{os.path.dirname(self.config.MODEL.PRETRAINED_DETECTOR_CKPT)}/recall.pt',map_location='cpu')
            logger.info(f'load previous predicate recall score success, recall info: {pre_step_res}')
            self.previou_rel_score=[pre_step_res[rel_name] if rel_name in pre_step_res.keys() else 1.0  for rel_name in self.rel_classes]
            self.high_conf_cls=(torch.tensor(self.previou_rel_score)>0.5).int()
        else:
            logger.warning('load previous recall score failed........')
            self.previou_rel_score=[0.0]*self.num_rel_cls
        
        self.previou_rel_score=torch.tensor(self.previou_rel_score)
                    
    def get_prior_reps(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),rel_nums=-1, **kwargs):
        if isinstance(sub_embeds,(list,tuple)):
            sub_embeds=torch.cat(sub_embeds,dim=0)
        if isinstance(obj_embeds,(list,tuple)):
            obj_embeds=torch.cat(obj_embeds,dim=0)

        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        
        if kwargs.get('predicate_proto',None) is not None and self.baseline_model=='PENet':
            proj_predicate_proto=self.project_prot_head(kwargs['predicate_proto'])
        else:
            predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
            proj_predicate_proto = self.project_prot_head(self.filter_pred_prot(predicate_proto))
            
        pair_preds,pair_feats=obj_infos['pair_pred'],obj_infos['pair_feat'] # pair_feats: fused roi features, semantic features and postion features
        
        # sub_sem_reps,obj_sem_reps=self.W_obj(self.obj_embed(pair_preds[:,0].long())),self.W_obj(self.obj_embed(pair_preds[:,1].long()))
        
        sub_node_feats,obj_node_feats=pair_feats[:,0,...],pair_feats[:,1,...]
        
        cps_union_reps,cps_t_sub_reps,cps_t_obj_reps=self.cps_union_reps(union_reps),self.cps_t_sub_reps(sub_embeds),self.cps_t_obj_reps(obj_embeds)
        
        # ---------------------- generate relation edge reps from subject-object ----------------------
        # node - node ==> interaction
        
        edg_rel_reps=self.edge_rel_reps.expand(cps_union_reps.shape[0],-1)
        for attn_sub_node,attn_obj_node,cs_ln,cs,mlp_ln,mlp,attn_rel_pro in self.node_to_pre:
            sub_node_feats=attn_sub_node(sub_node_feats,obj_node_feats,rel_nums)
            obj_node_feats=attn_obj_node(obj_node_feats,sub_node_feats,rel_nums)
            
            entity_pairs,edg_rel_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),edg_rel_reps.unsqueeze(1)
            edg_rel_reps_out,_=cs(query=edg_rel_reps,key=entity_pairs,value=entity_pairs)
            edg_rel_reps=cs_ln(edg_rel_reps+edg_rel_reps_out)
            
            edg_rel_reps=mlp_ln(mlp(edg_rel_reps)+edg_rel_reps)

            edg_rel_reps=attn_rel_pro(edg_rel_reps.squeeze(1),proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)

        for union_attn_entity,filter_entity,union_attn_prot,refine_edge_rel in self.refine_edge_pre:
            ln_cs,cs,ln_mlp,mlp = union_attn_entity

            entity_pairs,cps_union_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),cps_union_reps.unsqueeze(1)
            cps_union_reps_out,_ =cs(query=cps_union_reps,key=entity_pairs,value=entity_pairs)
            cps_union_reps_out=ln_cs(cps_union_reps+cps_union_reps_out)
            
            cps_union_reps_out=ln_mlp(mlp(cps_union_reps_out)+cps_union_reps_out)
            
            cps_union_reps_out,cps_union_reps=cps_union_reps_out.squeeze(1),cps_union_reps.squeeze(1)
            cps_union_reps=cps_union_reps-filter_entity(torch.cat([sub_node_feats,obj_node_feats],dim=-1))*cps_union_reps_out

            union_prot=union_attn_prot(cps_union_reps,proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)
            
            edg_rel_reps=refine_edge_rel(edg_rel_reps,union_prot,rel_nums)

        proj_edg_rel_reps=self.align_head(self.filter_noise_rel(edg_rel_reps))

        # ---------------------- init denoise module ----------------------
        # generate predicate reps based on triple
        cps_entity_pair_reps=self.cps_entity_pair_reps(torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1))
        cps_ctx_reps=cps_union_reps+cps_entity_pair_reps*cps_union_reps
        tri_rel_ctx_reps=cps_ctx_reps+cps_ctx_reps*self.gate_vis_entity(torch.cat([cps_ctx_reps,cps_entity_pair_reps],dim=-1))
        
        if self.use_denoise_branch:
            noise=torch.randn(tri_rel_ctx_reps.shape).to(device)
            noised_tri_rel_reps=tri_rel_ctx_reps+noise*self.noise_factor*tri_rel_ctx_reps
            
            for init_denoise,denoise_entity in self.denoise_modules:
                noised_tri_rel_reps=init_denoise(noised_tri_rel_reps)
                
                # ************************************************
                sub_atn_block,obj_atn_block,cps_entity_pair,rel_atn_entity,filter_entity,rel_atn_union=denoise_entity
                # attention subject features
                cps_t_sub_reps=sub_atn_block(cps_t_sub_reps,cps_t_sub_reps,rel_nums)
                
                # attention subject features
                cps_t_obj_reps=obj_atn_block(cps_t_obj_reps,cps_t_obj_reps,rel_nums)
                
                # filter subject-object features
                entity_embeds=torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1)
                cps_entity_embeds=cps_entity_pair(entity_embeds)
                noise_entity_out=rel_atn_entity(noised_tri_rel_reps,cps_entity_embeds,rel_nums)
                
                noised_tri_rel_reps=noised_tri_rel_reps-noise_entity_out*filter_entity(entity_embeds)
                
                # refine noised triple rel reps
                noised_tri_rel_reps=rel_atn_union(noised_tri_rel_reps,union_prot,rel_nums)
                
            denoise_tri_rel_reps=self.denoise_reps(noised_tri_rel_reps)
            
            proj_denoise_tri_rel_reps=self.align_head(self.filter_noise_rel(denoise_tri_rel_reps))
        else:
            proj_denoise_tri_rel_reps=None
        # ************ align predicate representation ************
        proj_pre_prot=self.align_head(proj_predicate_proto)
        
        if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
            sum_rel_reps=proj_edg_rel_reps*self.filter_tri(proj_edg_rel_reps)+proj_denoise_tri_rel_reps*self.filter_recon(proj_denoise_tri_rel_reps)
        else:
            sum_rel_reps=None
            
        # ************ using global features to refine local features ************
        if self.use_global_vis_refine:
            max_size=kwargs['enc_features'][-1].shape[-2:]
            enc_features=[F.interpolate(enc_rep,size=max_size,mode='bilinear',align_corners=False) for enc_rep in kwargs['enc_features']]  # list()
            enc_features=torch.cat(enc_features,dim=1).flatten(start_dim=2).permute(0,2,1).contiguous()
            enc_features=self.proj_glob_reps(self.filter_glob_reps(self.ds_glob_reps(enc_features)))
            
            for (refine_edg_module,refine_recon_module,refine_sum_module) in self.glob_refine_rel_reps:
                
                if self.use_node_branch:
                    proj_edg_rel_reps=refine_edg_module(proj_edg_rel_reps,kv_feats=enc_features,q_split=rel_nums)

                if self.use_denoise_branch:
                    proj_denoise_tri_rel_reps=refine_recon_module(proj_denoise_tri_rel_reps,kv_feats=enc_features,q_split=rel_nums)
                
                if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
                    sum_rel_reps=refine_sum_module(sum_rel_reps,kv_feats=enc_features,q_split=rel_nums)
        
        if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
            # *********** merge predicate reps ***********
            glob_rel_reps=self.global_rel_reps.unsqueeze(0).expand(sum(rel_nums),-1)
            all_rel_reps=torch.stack([sum_rel_reps,proj_denoise_tri_rel_reps,proj_edg_rel_reps],dim=1) if self.use_branch_fusion else torch.stack([proj_denoise_tri_rel_reps,proj_edg_rel_reps],dim=1)
            for merge_rel_module in self.merge_rel_reps:
                ln_sa_reps,sa_reps,ln_glob_reps_sa,glob_reps_sa,ln_ca,ca,ln_mlp,mlp=merge_rel_module
                
                all_rel_reps_attn_out,_=sa_reps(all_rel_reps,all_rel_reps,all_rel_reps)
                all_rel_reps=all_rel_reps+ln_sa_reps(all_rel_reps_attn_out)
                
                glob_rel_reps_attn_out,_=ca(glob_rel_reps.unsqueeze(1),all_rel_reps,all_rel_reps)
                glob_rel_reps=glob_rel_reps+ln_ca(glob_rel_reps_attn_out.squeeze(1))
                
                glob_rel_reps_attn_out,_=glob_reps_sa(glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0))
                glob_rel_reps=glob_rel_reps+ln_glob_reps_sa(glob_rel_reps_attn_out.squeeze(0))
                
                glob_rel_reps=glob_rel_reps+ln_mlp(mlp(glob_rel_reps))
        
        else:
            glob_rel_reps=None

        return (proj_edg_rel_reps,proj_denoise_tri_rel_reps,sum_rel_reps,glob_rel_reps,proj_pre_prot),add_losses        
    
    def forward(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),proposals=None,rel_pairs=None,rel_nums=-1, **kwargs):
        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        
        def cal_kl_div(mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
            if mu1 is None or logvar1 is None:
                KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
            else:
                KLD = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
            if norm_value is not None:
                KLD = KLD / float(norm_value)
            return KLD
        
        if self.step==1:
            pre_reps,add_losses=self.get_prior_reps(sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=rel_labels,add_losses=add_losses,rel_nums=rel_nums, **kwargs)
            
            edg_rel_reps,recon_tri_rel_reps,sum_rel_reps,glob_rel_reps,rel_proto=pre_reps
            
            rel_prot_norm = rel_proto / rel_proto.norm(dim=1, keepdim=True)
            
            reps_dict,predict_dict,overall_dict=dict(),dict(),dict()
            if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
                sum_rel_reps_norm=sum_rel_reps/sum_rel_reps.norm(dim=1,keepdim=True)
                sum_rel_sim=(sum_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                
                reps_dict['sum_rel_reps']=sum_rel_reps
                predict_dict['sum_rel_sim']=sum_rel_sim
                overall_dict['sum_branch']=dict(rel_reps=sum_rel_reps,rel_sim=sum_rel_sim)
                
            if self.use_denoise_branch:
                proj_denoise_tri_rel_norm = recon_tri_rel_reps / recon_tri_rel_reps.norm(dim=1, keepdim=True)
                denoise_rel_sim=(proj_denoise_tri_rel_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)

                reps_dict['recon_tri_rel_reps']=recon_tri_rel_reps
                predict_dict['recon_rel_sim']=denoise_rel_sim
                overall_dict['recon_branch']=dict(rel_reps=recon_tri_rel_reps,rel_sim=denoise_rel_sim)
                
            if self.use_node_branch:
                proj_edg_rel_reps_norm = edg_rel_reps / edg_rel_reps.norm(dim=1, keepdim=True)
                edg_rel_sim=(proj_edg_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)  

                reps_dict['edg_rel_reps']=edg_rel_reps
                predict_dict['edg_rel_sim']=edg_rel_sim
                overall_dict['edg_branch']=dict(rel_reps=edg_rel_reps,rel_sim=edg_rel_sim)
                
            if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
                glob_rel_reps_norm=glob_rel_reps/glob_rel_reps.norm(dim=1,keepdim=True)
                glob_rel_sim=(glob_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                
                reps_dict['glob_rel_reps']=glob_rel_reps
                predict_dict['glob_rel_sim']=glob_rel_sim
                overall_dict['glob_branch']=dict(rel_reps=glob_rel_reps,rel_sim=glob_rel_sim)
            
            if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
                vae_pre_dist,vae_proto,losses=self.VAE_decoder(glob_rel_reps,rel_proto,union_reps,rel_nums,rel_labels if self.training else None)
            else:
                raise ValueError('Not implement!!!')
            # *************** for step 1 ,to calculate the similar between predicate reps and prototype ***************
            
            if self.training:
                rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
                add_losses=self.overall_reps_loss(rel_proto,rel_prot_norm,overall_dict,device,rel_labels,add_losses)
                
                if self.use_kl_modules and not self.use_kl_weight_loss:
                    pre_proto_mean,pre_proto_logvar=self.kl_infos[0](rel_proto),self.kl_infos[1](rel_proto)
                    pre_proto_mean,pre_proto_logvar=pre_proto_mean[rel_labels],pre_proto_logvar[rel_labels]
                    
                    for name,reps in reps_dict.items():
                        rel_reps_mean,rel_reps_logvar=self.kl_infos[0](reps),self.kl_infos[1](reps)
                        add_losses[f'{name}_kl_loss']=add_losses.get(f'{name}_kl_loss',0.0)+cal_kl_div(rel_reps_mean,rel_reps_logvar,pre_proto_mean,pre_proto_logvar)
                
            if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
                final_pre=predict_dict['glob_rel_sim']
            else:
                final_pre=sum(predict_dict.values())
        
        else:
            with torch.no_grad():
                pre_reps,add_losses=self.get_prior_reps(sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=rel_labels,add_losses=add_losses,rel_nums=rel_nums, **kwargs)
                edg_rel_reps,recon_tri_rel_reps,sum_rel_reps,glob_rel_reps,rel_proto=pre_reps
                
                rel_prot_norm = rel_proto / rel_proto.norm(dim=1, keepdim=True)
            
                reps_dict,predict_dict=dict(),dict()
                if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
                    sum_rel_reps_norm=sum_rel_reps/sum_rel_reps.norm(dim=1,keepdim=True)
                    sum_rel_sim=(sum_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                    
                    reps_dict['sum_rel_reps']=sum_rel_reps
                    predict_dict['sum_rel_sim']=sum_rel_sim
                    
                if self.use_denoise_branch:
                    proj_denoise_tri_rel_norm = recon_tri_rel_reps / recon_tri_rel_reps.norm(dim=1, keepdim=True)
                    denoise_rel_sim=(proj_denoise_tri_rel_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)

                    reps_dict['recon_tri_rel_reps']=recon_tri_rel_reps
                    predict_dict['recon_rel_sim']=denoise_rel_sim
                    
                if self.use_node_branch:
                    proj_edg_rel_reps_norm = edg_rel_reps / edg_rel_reps.norm(dim=1, keepdim=True)
                    edg_rel_sim=(proj_edg_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)  

                    reps_dict['edg_rel_reps']=edg_rel_reps
                    predict_dict['edg_rel_sim']=edg_rel_sim
                    
                if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
                    glob_rel_reps_norm=glob_rel_reps/glob_rel_reps.norm(dim=1,keepdim=True)
                    glob_rel_sim=(glob_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                    
                    reps_dict['glob_rel_reps']=glob_rel_reps
                    predict_dict['glob_rel_sim']=glob_rel_sim
                
                if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
                    head_pre_dist=predict_dict['glob_rel_sim']
                    context_reps=reps_dict['glob_rel_reps']
                else:
                    head_pre_dist=sum(predict_dict.values())
                    context_reps=torch.cat([reps for reps in reps_dict.values()],dim=-1)
            
            if getattr(self,'fdm',None) is not None:
                if self.training:
                    final_pre,proto,losses=self.fdm(context_reps,rel_proto,union_reps,rel_nums,rel_labels)
                    add_losses.update(losses)
                else:
                    with torch.no_grad():
                        final_pre,_,_=self.fdm(context_reps,rel_proto,union_reps,rel_nums)
                        
            final_pre=final_pre+head_pre_dist
        torch.cuda.empty_cache()            
        return final_pre,dict(),add_losses
         
    def overall_reps_loss(self,rel_proto,rel_prot_norm,overall_dict,device,rel_labels,add_losses):
        if self.step==1:
            add_losses=self.init_proto_loss(rel_proto,rel_prot_norm,add_losses)
        
        self.gt_scores=self.gt_scores.to(device=device)+torch.sum(F.one_hot(rel_labels,self.num_rel_cls).to(device=device),dim=0)
        for name,rep_sim_dict in overall_dict.items():
            rel_reps,rel_sim=rep_sim_dict['rel_reps'],rep_sim_dict['rel_sim']
            
            if self.use_kl_weight_loss:
                kl_div=F.kl_div(rel_reps.softmax(dim=-1).log(),rel_proto.softmax(dim=-1)[rel_labels],reduction='none')
            
                rel_rep_ce_loss=F.cross_entropy(rel_sim,rel_labels,reduction='none')
                add_losses['sim_ce_loss']=add_losses.get('sim_ce_loss',0.0)+torch.mean(rel_rep_ce_loss*kl_div.sum(-1))
                
                add_losses=self.predicate_reps_loss(rel_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss',f'{name}_rep2proto_dist',kl_div=kl_div.sum(-1))
                
                kl_div_loss=torch.max(torch.zeros(kl_div.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),kl_div.sum(dim=-1)).mean()
                add_losses['kl_div_loss']=add_losses.get('kl_div_loss',0.0)+kl_div_loss
            elif self.use_adaptive_loss:
                add_losses=self.predicate_reps_loss(rel_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss',f'{name}_rep2proto_dist')
                if 'glob_branch' in name:
                    continue
                with torch.no_grad():
                    pos_mask=torch.zeros(rel_sim.shape,device=device)
                    pos_mask[torch.arange(rel_sim.shape[0]),rel_labels]=1

                    pos_scores=getattr(self,f'pos_{name}_scores').to(device)+torch.sum(pos_mask.long()*rel_sim,dim=0)
                    setattr(self,f'pos_{name}_scores',pos_scores)
                    
                    neg_scores=getattr(self,f'neg_{name}_scores').to(device)+torch.sum((1-pos_mask.long())*rel_sim,dim=0)
                    setattr(self,f'neg_{name}_scores',neg_scores)
                    
                    emp_weight=self.emp_decay*getattr(self,f'{name}_emp_weight').to(device)+(1-self.emp_decay)*torch.log(1+(neg_scores/(self.gt_scores+1e-5))/(pos_scores/(self.gt_scores+1e-5)+1e-5))
                    emp_weight[0]=1e-5
                    setattr(self,f'{name}_emp_weight',emp_weight)
                    
                add_losses[f'{name}_adaptive_loss']=add_losses.get(f'{name}_adaptive_loss',0.0)+F.cross_entropy(rel_sim,rel_labels,weight=emp_weight)
            else:
                add_losses=self.predicate_reps_loss(rel_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss',f'{name}_rep2proto_dist')
                add_losses[f'{name}_loss']=add_losses.get(f'{name}_loss',0.0)+F.cross_entropy(rel_sim,rel_labels)
            
        return add_losses
    
    def compute_gradient_penalty(self, module, real_samples, fake_samples):
        alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = module(interpolates)
        fake = torch.ones(d_interpolates.shape, device=real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def init_proto_loss(self,predicate_proto,predicate_proto_norm,add_losses):
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
        simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
        add_losses['l21_loss']=add_losses.get('l21_loss',0.0)+l21  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
        predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses['dist_loss2']=add_losses.get('dist_loss2',0.0)+dist_loss
        ### end 
        return add_losses
    
    def predicate_reps_loss(self,rel_reps,rel_center,rel_labels,add_losses,loss_fun,loss_name,kl_div=None,extra_weight=1.0):
        if isinstance(rel_labels,(list,tuple)):
            rel_labels=torch.cat(rel_labels,dim=0)
        if 'intra_cls_loss' in loss_fun:
            assert rel_labels!=None,'Please check relation labels!'
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],self.num_rel_cls,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),rel_labels]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),rel_labels]
            if kl_div is None:
                kl_div=torch.ones_like(pos_dis,device=pos_dis.device)
            pos_dis=pos_dis*kl_div
            
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss*extra_weight
        
        if 'inter_cls_loss' in loss_fun:
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,rel_center.shape[0],-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],rel_center.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss*extra_weight

        return add_losses

    def calculate_loss(self,relation_logits,rel_labels,proposals=None,refine_logits=None):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        if proposals is not None and refine_logits is not None:
            refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
            fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            
            obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        else:
            obj_loss= None
        # ********************************************************************
        
        return rel_loss,obj_loss
   