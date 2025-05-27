# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import functools
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.abspath(os.path.join(current_dir,'../')))
from glob import glob
import argparse
import time
import torch
from tqdm import tqdm
from tools.relation_train_net import fix_eval_modules
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.comm import synchronize
import numpy
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from openpyxl import Workbook

head=['on','has','wearing','of','in','near','behind']
body=['with','holding','above','under','wears','sitting on','in front of','riding','standing on','at',
        'attached to','over','carrying','walking on','for','looking at','watching','hanging from','belonging to',
        'and','parked on']
tail=['between','laying on','along','eating','covering','covered in','part of','using','to','on back of',
        'across','mounted on','lying on','walking in','against','from','growing on','painted on','made of',
        'playing','says','flying in']
idx_to_predicate={"1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
    

def load_reps():
    sample_ids=os.listdir("{base path}")
    dpplml_path,pe_path='{base path}','{base path}'
    
    pe_sub_emb,pe_obj_emb,pe_rel_reps,pe_rel_labels=[],[],[],[]
    
    dpplml_sub_emb,dpplml_obj_emb,dpplml_entity_rel_rep=[],[],[]
    dpplml_s_p_rep,dpplml_o_p_rep,dpplml_rel_rep,dpplml_rel_center,dpplml_rel_labels=[],[],[],[],[]
    # for sample_id in tqdm(random.sample(sample_ids,k=1000)):
    for sample_id in tqdm(sample_ids):
        # ************ *************** ************
        dpplml_entity_dicts=torch.load(f'{dpplml_path}/{sample_id}/entity_reps.pth',map_location='cpu')
        dpplml_rel_dicts=torch.load(f'{dpplml_path}/{sample_id}/rel_reps.pth',map_location='cpu')
        
        dpplml_labels=dpplml_rel_dicts['rel_label'].cpu()
        
        # ************ PE Net Features ************
        pe_entity_dicts=torch.load(f'{pe_path}/{sample_id}/entity_features.pth',map_location='cpu')
        pe_rel_dicts=torch.load(f'{pe_path}/{sample_id}/rel_features.pth',map_location='cpu')
        
        pe_labels=pe_rel_dicts['rel_labels'].cpu()
        
        # ************ *************** ************
        # continue_epoch=False
        assert len(pe_labels)==len(dpplml_labels)
        
        # ************ insert data ************
        
        dpplml_sub_emb.append(dpplml_entity_dicts['sub_embeds'].cpu())
        dpplml_obj_emb.append(dpplml_entity_dicts['obj_embeds'].cpu())
        dpplml_entity_rel_rep.append(dpplml_entity_dicts['rel_reps'])
        
        dpplml_s_p_rep.append(dpplml_rel_dicts['sp_query'].cpu())
        dpplml_o_p_rep.append(dpplml_rel_dicts['op_query'].cpu())
        dpplml_rel_rep.append(dpplml_rel_dicts['rel_query'].cpu())
        dpplml_rel_center.append(dpplml_rel_dicts['rel_center'].cpu())  # all_samples,num_rels,hidden_dim
        dpplml_pro=dpplml_rel_dicts['predicate_prototye'].cpu() # num_rels,hidden_dim
        dpplml_rel_labels.append(dpplml_labels)
        
        pe_sub_emb.append(pe_entity_dicts['sub_sem_reps'].cpu())
        pe_obj_emb.append(pe_entity_dicts['obj_sem_reps'].cpu())
        pe_pro=pe_rel_dicts['predicate_proto'].cpu()
        
        pe_rel_reps.append(pe_rel_dicts['rel_sem_reps'].cpu())
        pe_rel_labels.append(pe_labels)
        
        null_cls=False
        for rel_id in range(1,51):
            exist_labels=torch.cat(dpplml_rel_labels,dim=0)
            if torch.sum(exist_labels==rel_id)<30:
                null_cls=True
                break
        
        if not null_cls:
            print(f'Each class have features more than fifty.')
            break
             
    
    return torch.cat(dpplml_sub_emb,dim=0),torch.cat(dpplml_obj_emb,dim=0),torch.cat(dpplml_entity_rel_rep,dim=0),torch.cat(dpplml_s_p_rep,dim=0),torch.cat(dpplml_o_p_rep,dim=0),torch.cat(dpplml_rel_rep,dim=0),torch.cat(dpplml_rel_center,dim=0),dpplml_pro,torch.cat(dpplml_rel_labels,dim=0),torch.cat(pe_sub_emb,dim=0),torch.cat(pe_obj_emb,dim=0),torch.cat(pe_rel_reps,dim=0),pe_pro,torch.cat(pe_rel_labels,dim=0)

def load_reps_for_nodis(base_name,base_path='{base path}/reps_space'):
    sample_ids=os.listdir(base_path)
    
    ori,step1,step2=dict(),dict(),dict()
    ori_save_nums,step1_save_nums,step2_save_nums=dict(),dict(),dict()
    for i in range(1,51):
        ori_save_nums[i]=0
        step1_save_nums[i]=0
        step2_save_nums[i]=0
    # for sample_id in tqdm(random.sample(sample_ids,k=10000)):
    for sample_id in tqdm(sample_ids):
        if not os.path.isdir(f'{base_path}/{sample_id}'):
            continue
        # ************ *************** ************
        if os.path.exists(f'{base_path}/{sample_id}/{base_name}_None_step_1.pt'):
            none_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_None_step_1.pt',map_location='cpu')
        else:
            raise ValueError(f'{base_path}/{sample_id}/{base_name}_None_step_1.pt is not found!!')
        
        if "PENet" in base_name:
            if os.path.exists(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt'):
                step1_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_1.pt',map_location='cpu')
            else:
                raise ValueError(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_1.pt is not found!!')
            
            if os.path.exists(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_2.pt'):
                step2_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_2.pt',map_location='cpu')
            else:
                step2_infos=None
        else:
            if os.path.exists(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt'):
                step1_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt',map_location='cpu')
            else:
                raise ValueError(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt is not found!!')
            
        #     if os.path.exists(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_2.pt'):
        #         step2_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_2.pt',map_location='cpu')
        #     else:
        #         step2_infos=None
                
        # if step2_infos is None:
        #     print(f'{sample_id}_Multi_step_Denoise_step_2 is not exits')
        #     continue
        
        for rel_id in range(1,51):
            if ori_save_nums[rel_id]<10000:
                save_info,save_len=load_transformer_reps(rel_id,none_infos,ori,'None')
                if save_info is not None:
                    ori=save_info
                    ori_save_nums[rel_id]=ori_save_nums[rel_id]+save_len
                
            min_idx=min(list(ori_save_nums.values()))//1000
            if min_idx>=1:
                if not os.path.exists(f'{base_path}/{base_name}_None_{min_idx}.pt'):
                    torch.save(dict(infos=ori,nums=ori_save_nums),f'{base_path}/{base_name}_None_{min_idx}.pt')
                    print(f'save {base_name}_None_{min_idx} predict infos success, min len: {min(list(ori_save_nums.values()))}.....')
            
            if step1_save_nums[rel_id]<10000:
                save_info,save_len=load_transformer_reps(rel_id,step1_infos,step1,'step1')
                if save_info is not None:
                    step1=save_info
                    step1_save_nums[rel_id]=step1_save_nums[rel_id]+save_len
                
            min_idx=min(list(step1_save_nums.values()))//1000
            if min_idx>=1:
                if not os.path.exists(f'{base_path}/{base_name}_step1_{min_idx}.pt'):
                    torch.save(dict(infos=step1,nums=step1_save_nums),f'{base_path}/{base_name}_step1_{min_idx}.pt')
                    print(f'save {base_name}_step1_{min_idx} predict infos success, min len: {min(list(step1_save_nums.values()))}.....')
                    
            # if step2_save_nums[rel_id]<10000:
            #     try:
            #         save_info,save_len=load_transformer_reps(rel_id,step2_infos,step2,'step2')
            #     except:
            #         save_info,save_len=None,0
            #     if save_info is not None:
            #         step2=save_info
            #         step2_save_nums[rel_id]=step2_save_nums[rel_id]+save_len
                    
            # min_idx=min(list(step2_save_nums.values()))//1000
            # if min_idx>=1:
            #     if not os.path.exists(f'{base_path}/{base_name}_step2_{min_idx}.pt'):
            #         torch.save(dict(infos=step2,nums=step2_save_nums),f'{base_path}/{base_name}_step2_{min_idx}.pt')
            #         print(f'save {base_name}_step2_{min_idx} predict infos success, min len: {min(list(step2_save_nums.values()))}.....')
            
    torch.save(dict(infos=ori,nums=ori_save_nums),f'{base_path}/{base_name}_None.pt')
    print(f'save {base_name}_None predict infos success.....')
    torch.save(dict(infos=step1,nums=step1_save_nums),f'{base_path}/{base_name}_step1.pt')
    print(f'save {base_name}_step1 predict infos success.....')
    # torch.save(dict(infos=step2,nums=step2_save_nums),f'{base_path}/{base_name}_step2.pt')
    # print(f'save {base_name}_step2 predict infos success.....')
    
             
def load_transformer_reps(rel_id,load_infos,save_infos,types='None'):
    sub_,obj_,prod_,vis_,label=load_infos['overall']['sub_embed'].cpu(),load_infos['overall']['obj_embeds'].cpu(),load_infos['overall']['prod_rep'].cpu(),load_infos['overall']['vis_rep'].cpu(),load_infos['overall']['rel_label'].cpu()
    # if len(torch.where(label==rel_id))<5:
    #     return None,0
    sample_idx=torch.where(label==rel_id)
    sub_,obj_,prod_,vis_=sub_[sample_idx],obj_[sample_idx],prod_[sample_idx],vis_[sample_idx]
    if rel_id not in save_infos:
        this_rel_infos=dict(sub_embeds=sub_,obj_embeds=obj_,prod_rep=prod_,vis_rep=vis_)
    else:
        this_rel_infos=save_infos[rel_id]
        this_rel_infos['sub_embeds']=torch.cat([this_rel_infos['sub_embeds'],sub_],dim=0)
        this_rel_infos['obj_embeds']=torch.cat([this_rel_infos['obj_embeds'],obj_],dim=0)
        this_rel_infos['prod_rep']=torch.cat([this_rel_infos['prod_rep'],prod_],dim=0)
        this_rel_infos['vis_rep']=torch.cat([this_rel_infos['vis_rep'],vis_],dim=0)
    
    if types=='step1':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
    if types=='step2':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
        this_rel_infos=load_step2_infos(sample_idx,load_infos,this_rel_infos)
    
    save_infos[rel_id]=this_rel_infos    
    return save_infos,len(torch.where(label==rel_id)) 

def load_penet_reps(rel_id,load_infos,save_infos,types='None'):
    sub_,obj_,rep_,proto_,label,fusion_=load_infos['overall']['sub_embed'].cpu(),load_infos['overall']['obj_embeds'].cpu(),load_infos['overall']['rel_rep'].cpu(),load_infos['overall']['predicate_proto'].cpu(),load_infos['overall']['rel_label'].cpu(),load_infos['overall']['fusion_so'].cpu()
    # if len(torch.where(label==rel_id))<5:
    #     return None,0
    sample_idx=torch.where(label==rel_id)
    sub_,obj_,rep_,proto_,fusion_=sub_[sample_idx],obj_[sample_idx],rep_[sample_idx],proto_[rel_id],fusion_[sample_idx]
    if rel_id not in save_infos:
        this_rel_infos=dict(sub_embeds=sub_,obj_embeds=obj_,rel_reps=rep_,fusion_so=fusion_,proto=proto_)
    else:
        this_rel_infos=save_infos[rel_id]
        this_rel_infos['sub_embeds']=torch.cat([this_rel_infos['sub_embeds'],sub_],dim=0)
        this_rel_infos['obj_embeds']=torch.cat([this_rel_infos['obj_embeds'],obj_],dim=0)
        this_rel_infos['rel_reps']=torch.cat([this_rel_infos['rel_reps'],rep_],dim=0)
        this_rel_infos['fusion_so']=torch.cat([this_rel_infos['fusion_so'],fusion_],dim=0)
    
    if types=='step1':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
    if types=='step2':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
        this_rel_infos=load_step2_infos(sample_idx,load_infos,this_rel_infos)
    
    save_infos[rel_id]=this_rel_infos    
    return save_infos,len(torch.where(label==rel_id))
    
def load_step1_infos(sample_idx,load_infos,this_rel_infos):
    sub_node,obj_node,refine_edg_rel_rep,filter_denoised_edg_rel_rep,proj_edg_rel_reps,noised_tri_rel_rep,denoise_tri_rel_rep,proj_denoise_tri_rel_rep,filter_denoised_tri_rel_rep,sum_rel_rep,glob_rel_rep=load_infos['step1']['sub_node'],load_infos['step1']['obj_node'],load_infos['step1']['refine_edg_rel_rep'],load_infos['step1']['filter_denoised_edg_rel_rep'],load_infos['step1']['proj_edg_rel_reps'],load_infos['step1']['noised_tri_rel_rep'],load_infos['step1']['denoise_tri_rel_rep'],load_infos['step1']['proj_denoise_tri_rel_rep'],load_infos['step1']['filter_denoised_tri_rel_rep'],load_infos['step1']['sum_rel_rep'],load_infos['step1']['glob_rel_rep']
    
    if 'sub_nodes' not in this_rel_infos.keys():
        this_rel_infos['sub_nodes']=sub_node[sample_idx]
    else:
        this_rel_infos['sub_nodes']=torch.cat([this_rel_infos['sub_nodes'],sub_node[sample_idx]],dim=0)
    if 'obj_nodes' not in this_rel_infos.keys():
        this_rel_infos['obj_nodes']=obj_node[sample_idx]
    else:
        this_rel_infos['obj_nodes']=torch.cat([this_rel_infos['obj_nodes'],obj_node[sample_idx]],dim=0) 
    if 'refine_edg_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['refine_edg_rel_reps']=refine_edg_rel_rep[sample_idx]
    else:
        this_rel_infos['refine_edg_rel_reps']=torch.cat([this_rel_infos['refine_edg_rel_reps'],refine_edg_rel_rep[sample_idx]],dim=0)
    if 'filter_denoised_edg_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['filter_denoised_edg_rel_reps']=filter_denoised_edg_rel_rep[sample_idx]
    else:
        this_rel_infos['filter_denoised_edg_rel_reps']=torch.cat([this_rel_infos['filter_denoised_edg_rel_reps'],filter_denoised_edg_rel_rep[sample_idx]],dim=0)
    if 'proj_edg_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['proj_edg_rel_reps']=proj_edg_rel_reps[sample_idx]
    else:
        this_rel_infos['proj_edg_rel_reps']=torch.cat([this_rel_infos['proj_edg_rel_reps'],proj_edg_rel_reps[sample_idx]],dim=0)
    if 'noised_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['noised_tri_rel_reps']=noised_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['noised_tri_rel_reps']=torch.cat([this_rel_infos['noised_tri_rel_reps'],noised_tri_rel_rep[sample_idx]],dim=0)
    if 'denoise_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['denoise_tri_rel_reps']=denoise_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['denoise_tri_rel_reps']=torch.cat([this_rel_infos['denoise_tri_rel_reps'],denoise_tri_rel_rep[sample_idx]],dim=0)
    if 'proj_denoise_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['proj_denoise_tri_rel_reps']=proj_denoise_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['proj_denoise_tri_rel_reps']=torch.cat([this_rel_infos['proj_denoise_tri_rel_reps'],proj_denoise_tri_rel_rep[sample_idx]],dim=0)
    if 'filter_denoised_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['filter_denoised_tri_rel_reps']=filter_denoised_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['filter_denoised_tri_rel_reps']=torch.cat([this_rel_infos['filter_denoised_tri_rel_reps'],filter_denoised_tri_rel_rep[sample_idx]],dim=0)
    if 'sum_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['sum_rel_reps']=sum_rel_rep[sample_idx]
    else:
        this_rel_infos['sum_rel_reps']=torch.cat([this_rel_infos['sum_rel_reps'],sum_rel_rep[sample_idx]],dim=0)
    if 'glob_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['glob_rel_reps']=glob_rel_rep[sample_idx]
    else:
        this_rel_infos['glob_rel_reps']=torch.cat([this_rel_infos['glob_rel_reps'],glob_rel_rep[sample_idx]],dim=0)

    if 'step1_proto' not in this_rel_infos.keys() and 'rel_proto' in load_infos['step1'].keys():
        this_rel_infos['step1_proto']=load_infos['step1']['rel_proto']
    return this_rel_infos

def load_step2_infos(sample_idx,load_infos,this_rel_infos):
    encode_ctx,mean,std,x_T,recon_q_ctx,qu_proj_ctx,decode_ctx=load_infos['step2']['encode_ctx'],load_infos['step2']['mean'],load_infos['step2']['std'],load_infos['step2']['x_T'],load_infos['step2']['recon_q_ctx'],load_infos['step2']['qu_proj_ctx'],load_infos['step2']['decode_ctx']
    
    if 'encode_ctx' not in this_rel_infos.keys():
        this_rel_infos['encode_ctx']=encode_ctx[sample_idx]
    else:
        this_rel_infos['encode_ctx']=torch.cat([this_rel_infos['encode_ctx'],encode_ctx[sample_idx]],dim=0)
    if 'mean' not in this_rel_infos.keys():
        this_rel_infos['mean']=mean[sample_idx]
    else:
        this_rel_infos['mean']=torch.cat([this_rel_infos['mean'],mean[sample_idx]],dim=0) 
    if 'std' not in this_rel_infos.keys():
        this_rel_infos['std']=std[sample_idx]
    else:
        this_rel_infos['std']=torch.cat([this_rel_infos['std'],std[sample_idx]],dim=0)
    if 'x_T' not in this_rel_infos.keys():
        this_rel_infos['x_T']=x_T[sample_idx]
    else:
        this_rel_infos['x_T']=torch.cat([this_rel_infos['x_T'],x_T[sample_idx]],dim=0)
    if 'recon_q_ctx' not in this_rel_infos.keys():
        this_rel_infos['recon_q_ctx']=recon_q_ctx[sample_idx]
    else:
        this_rel_infos['recon_q_ctx']=torch.cat([this_rel_infos['recon_q_ctx'],recon_q_ctx[sample_idx]],dim=0)
    if 'qu_proj_ctx' not in this_rel_infos.keys():
        this_rel_infos['qu_proj_ctx']=qu_proj_ctx[sample_idx]
    else:
        this_rel_infos['qu_proj_ctx']=torch.cat([this_rel_infos['qu_proj_ctx'],qu_proj_ctx[sample_idx]],dim=0)
    if 'decode_ctx' not in this_rel_infos.keys():
        this_rel_infos['decode_ctx']=decode_ctx[sample_idx]
    else:
        this_rel_infos['decode_ctx']=torch.cat([this_rel_infos['decode_ctx'],decode_ctx[sample_idx]],dim=0)
    
    if 'step2_encode_proto' not in this_rel_infos.keys():
        this_rel_infos['step2_encode_proto']=load_infos['step2']['encode_proto']
    
    if 'step2_decode_proto' not in this_rel_infos.keys():
        this_rel_infos['step2_decode_proto']=load_infos['step2']['decode_proto']
    
    if 'q_embed' not in this_rel_infos.keys():
        this_rel_infos['q_embed']=load_infos['step2']['q_embed']
    
    return this_rel_infos
    
def var(reps,rel_labels,proto=None):
    if proto is not None:
        assert len(reps)==proto.shape[0]
    var_protos,var_reps=dict(),dict()

    for idx,(rep,rel_id) in enumerate(zip(reps,rel_labels)):
        if len(rep)==0:
            continue
        if proto is not None:
            variance_to_prototype = torch.mean(torch.sum((rep - proto[idx]) ** 2, dim=1)).item()
            var_protos[idx_to_predicate[str(rel_id)]]=variance_to_prototype
        
        feature_mean = torch.mean(rep, dim=0)  # (d,)
        variance_within_feature = torch.mean(torch.sum((rep - feature_mean) ** 2, dim=1)).item()
        var_reps[idx_to_predicate[str(rel_id)]]=variance_within_feature
    return var_protos,var_reps

def generate_excel(worksheet,data_dict,key_name,lg_list,index_name='B'):
    worksheet[f'{index_name}1']=key_name
    
    for idx,name in enumerate(lg_list,start=2):
        if index_name=='B':
            worksheet[f'A{idx}'] = name  
    
        worksheet[f'{index_name}{idx}'] = data_dict.get(name,0.0)
    
    return worksheet

def get_transformer_reps(choice_nums=30000):
    load_none_infos=torch.load('{base path}/reps_space/TransformerPredictor_None.pt',map_location='cpu')
    load_nodis_infos=torch.load('{base path}/reps_space/TransformerPredictor_step2.pt',map_location='cpu')
    non_sub_embeds,non_obj_embeds,non_vis_reps,non_prod_reps,rel_ids=[],[],[],[],[]
    nodis_sub_embeds,nodis_obj_embeds,nodis_vis_reps,nodis_prod_reps=[],[],[],[]
    
    nodis_glob_reps,nodis_encode_ctx,nodis_mean,nodis_std,nodis_xT,nodis_recon_xT,nodis_dis_ctx,nodis_decode_ctx=[],[],[],[],[],[],[],[]
    
    nodis_step1_proto,nodis_step2_enc_proto,nodis_step2_de_proto,nodis_step2_qembed=None,None,None,None
    
    sample_nums=0
    for rep_cls_id,none_rep_info in tqdm(load_none_infos['infos'].items()):
        nodis_rep_info=load_nodis_infos['infos'][rep_cls_id]
        
        if nodis_step1_proto is None:
            nodis_step1_proto=nodis_rep_info['step1_proto'][1:]
        if nodis_step2_enc_proto is None:
            nodis_step2_enc_proto=nodis_rep_info['step2_encode_proto'][1:]
        if nodis_step2_de_proto is None:
            nodis_step2_de_proto=nodis_rep_info['step2_decode_proto'][1:]
        
        if nodis_step2_qembed is None:
            nodis_step2_qembed=nodis_rep_info['q_embed'][1:]
            
        min_reps_num=min(none_rep_info['sub_embeds'].shape[0],nodis_rep_info['sub_embeds'].shape[0],choice_nums)
        sample_idx=torch.randperm(min_reps_num)
        sample_nums+=min_reps_num
        
        non_sub_embeds.append(none_rep_info['sub_embeds'][sample_idx])
        non_obj_embeds.append(none_rep_info['obj_embeds'][sample_idx])
        non_prod_reps.append(none_rep_info['prod_rep'][sample_idx])
        non_vis_reps.append(none_rep_info['vis_rep'][sample_idx])
        rel_ids.append(rep_cls_id)
        
        nodis_sub_embeds.append(nodis_rep_info['sub_embeds'][sample_idx])
        nodis_obj_embeds.append(nodis_rep_info['obj_embeds'][sample_idx])
        nodis_vis_reps.append(nodis_rep_info['vis_rep'][sample_idx])
        nodis_prod_reps.append(nodis_rep_info['prod_rep'][sample_idx])

        nodis_glob_reps.append(nodis_rep_info['glob_rel_reps'][sample_idx])
        nodis_encode_ctx.append(nodis_rep_info['encode_ctx'][sample_idx])
        nodis_xT.append(nodis_rep_info['x_T'][sample_idx])
        nodis_recon_xT.append(nodis_rep_info['recon_q_ctx'][sample_idx])
        nodis_dis_ctx.append(nodis_rep_info['qu_proj_ctx'][sample_idx])
        nodis_decode_ctx.append(nodis_rep_info['decode_ctx'][sample_idx])
    
    non_trans_data=torch.load("outputs/VG/None/TransformerPredictor_predcls_wo_bias_step1/best.pth",map_location="cpu")
    non_trans_rel_proto=non_trans_data['model']['module.roi_heads.relation.predictor.rel_compress.weight'].cpu()[1:]
    non_trans_ctx_proto=non_trans_data['model']['module.roi_heads.relation.predictor.ctx_compress.weight'].cpu()[1:]
    
    nodis_trans_data=torch.load("outputs/VG/Multi_step_Denoise/TransformerPredictor_predcls_wo_bias_step2/best.pth",map_location='cpu')
    nodis_trans_rel_proto=nodis_trans_data['model']['module.roi_heads.relation.predictor.rel_compress.weight'].cpu()[1:]
    nodis_trans_ctx_proto=nodis_trans_data['model']['module.roi_heads.relation.predictor.ctx_compress.weight'].cpu()[1:]
    
    lg_list=[]
    lg_list.extend(head)
    lg_list.extend(body)
    lg_list.extend(tail)
    
    inter_var_workbook = Workbook()
    proto_var_workbook = Workbook()
    inter_var_worksheet = inter_var_workbook.active
    proto_var_worksheet = proto_var_workbook.active
    inter_var_worksheet.title = "Sheet1" 
    proto_var_worksheet.title = "Sheet1" 

    inter_var_worksheet['A1']="predicate name"
    proto_var_worksheet['A1']="predicate name"
    print(f'=================== NoDIS, sample nums: {sample_nums} ===================')
    # pair_rep_scatter(pe_recon_xT,pe_dis_ctx,sample_nums,rel_ids,'nodis_2_recon',pca=False)
    var_protos,var_reps=var(nodis_glob_reps,rel_ids,nodis_step1_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'glob_reps_var',lg_list,index_name='B')
    generate_excel(inter_var_worksheet,var_reps,'glob_reps_var',lg_list,index_name='B')
    print(f'before diffusion glob proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_encode_ctx,rel_ids,nodis_step2_enc_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'align_reps',lg_list,index_name='C')
    generate_excel(inter_var_worksheet,var_reps,'align_reps',lg_list,index_name='C')
    print(f'before diffusion align encode proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_recon_xT,rel_ids,None)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'recon_reps_var',lg_list,index_name='D')
    generate_excel(inter_var_worksheet,var_reps,'recon_reps_var',lg_list,index_name='D')
    print(f'after diffusion recon ctx var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_dis_ctx,rel_ids,nodis_step2_qembed)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'discrete_reps_var',lg_list,index_name='E')
    generate_excel(inter_var_worksheet,var_reps,'discrete_reps_var',lg_list,index_name='E')
    print(f'after diffusion discrete recon proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_decode_ctx,rel_ids,nodis_step2_de_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'decode_reps_var',lg_list,index_name='F')
    generate_excel(inter_var_worksheet,var_reps,'decode_reps_var',lg_list,index_name='F')
    print(f'after diffusion discrete recon decoder proj proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_vis_reps,rel_ids,nodis_trans_rel_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'final_vis_reps_var',lg_list,index_name='G')
    generate_excel(inter_var_worksheet,var_reps,'final_vis_reps_var',lg_list,index_name='G')
    print(f'final vis rel proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_prod_reps,rel_ids,nodis_trans_ctx_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'final_ctx_reps_var',lg_list,index_name='H')
    generate_excel(inter_var_worksheet,var_reps,'final_ctx_reps_var',lg_list,index_name='H')
    print(f'final ctx proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    print('=================== wo NoDIS ===================')
    var_protos,var_reps=var(non_vis_reps,rel_ids,non_trans_rel_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'None_vis_reps_var',lg_list,index_name='I')
    generate_excel(inter_var_worksheet,var_reps,'None_vis_reps_var',lg_list,index_name='I')
    print(f'final vis rel proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(non_prod_reps,rel_ids,non_trans_ctx_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'None_ctx_reps_var',lg_list,index_name='J')
    generate_excel(inter_var_worksheet,var_reps,'None_ctx_reps_var',lg_list,index_name='J')
    print(f'final ctx proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    inter_var_workbook.save('Transformer_inter_var.xlsx')
    proto_var_workbook.save('Transformer_proto_var.xlsx')


def get_penet_reps(choice_nums=30000):
    load_none_infos=torch.load('{base path}/reps_space/PENetPredictor_None_10.pt',map_location='cpu')
    load_pe_infos=torch.load('{base path}/reps_space/PENetPredictor_step2_10.pt',map_location='cpu')
    
    non_sub_embeds,non_obj_embeds,non_rel_reps,non_fusion_so,non_proto,rel_ids=[],[],[],[],[],[]
    pe_sub_embeds,pe_obj_embeds,pe_rel_reps,pe_fusion_so,pe_proto=[],[],[],[],[]
    pe_glob_reps,pe_encode_ctx,pe_mean,pe_std,pe_xT,pe_recon_xT,pe_dis_ctx,pe_decode_ctx=[],[],[],[],[],[],[],[]
    
    pe_step1_proto,pe_step2_enc_proto,pe_step2_de_proto,step2_qembed=None,None,None,None
    
    sample_nums=0
    for rep_cls_id,none_rep_info in tqdm(load_none_infos['infos'].items()):
        pe_rep_info=load_pe_infos['infos'][rep_cls_id]
        
        if pe_step1_proto is None:
            pe_step1_proto=pe_rep_info['step1_proto'][1:]
        if pe_step2_enc_proto is None:
            pe_step2_enc_proto=pe_rep_info['step2_encode_proto'][1:]
        if pe_step2_de_proto is None:
            pe_step2_de_proto=pe_rep_info['step2_decode_proto'][1:]
        
        if step2_qembed is None:
            tmp_info=torch.load('{base path}/reps_space/2336708/PENetPredictor_v2_Multi_step_Denoise_step_2.pt',map_location='cpu')['step2']
            step2_qembed=tmp_info['q_embed'][1:]
            
        min_reps_num=min(none_rep_info['sub_embeds'].shape[0],pe_rep_info['sub_embeds'].shape[0],choice_nums)
        sample_idx=torch.randperm(min_reps_num)
        sample_nums+=min_reps_num
        
        non_sub_embeds.append(none_rep_info['sub_embeds'][sample_idx])
        non_obj_embeds.append(none_rep_info['obj_embeds'][sample_idx])
        non_rel_reps.append(none_rep_info['rel_reps'][sample_idx])
        non_fusion_so.append(none_rep_info['fusion_so'][sample_idx])
        non_proto.append(none_rep_info['proto'])
        rel_ids.append(rep_cls_id)
        
        pe_sub_embeds.append(pe_rep_info['sub_embeds'][sample_idx])
        pe_obj_embeds.append(pe_rep_info['obj_embeds'][sample_idx])
        pe_rel_reps.append(pe_rep_info['rel_reps'][sample_idx])
        pe_fusion_so.append(pe_rep_info['fusion_so'][sample_idx])
        pe_proto.append(pe_rep_info['proto'])

        pe_glob_reps.append(pe_rep_info['glob_rel_reps'][sample_idx])
        pe_encode_ctx.append(pe_rep_info['encode_ctx'][sample_idx])
        pe_xT.append(pe_rep_info['x_T'][sample_idx])
        pe_recon_xT.append(pe_rep_info['recon_q_ctx'][sample_idx])
        pe_dis_ctx.append(pe_rep_info['qu_proj_ctx'][sample_idx])
        pe_decode_ctx.append(pe_rep_info['decode_ctx'][sample_idx])
        
    non_proto, pe_proto=torch.stack(non_proto,dim=0),torch.stack(pe_proto,dim=0)

if __name__ == "__main__":
    # draw_bar_plot(['Original','After diffusion enhancement','After discretization mapping'],[0.435944351614738,1.58281202705539,0.0019520692344470053])
    # load_reps_for_nodis(base_name='TransformerPredictor')
    save_path='{base path}/quality_vis_reps'
    os.makedirs(save_path,exist_ok=True)
    
    get_penet_reps()
    # get_transformer_reps()