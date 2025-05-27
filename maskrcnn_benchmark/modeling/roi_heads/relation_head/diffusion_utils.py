import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.relation_head.attention_blocks import MLP, Trans_block
from maskrcnn_benchmark.modeling.make_layers import make_fc


class CouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim, swap=False):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def forward(self, x, logpx=None, reverse=False):

        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp
        
        
class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

  

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
    
class ControlDiff(nn.Module):
    def __init__(self,cfg,in_dim,num_steps):
        super().__init__()
        
        inner_dim=in_dim//2
        self.inner_dim=inner_dim
        hidden_dim=in_dim//4
        
        self.time_embedding=nn.Embedding(num_steps+1,inner_dim)
        nn.init.normal_(self.time_embedding.weight, mean=0, std=1)
        
        self.cond_query=nn.Parameter(torch.normal(mean=0, std=0.1, size=(inner_dim,)))
        self.align_cond=MLP(cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM,hidden_dim,inner_dim,2)
        self.align_ctx=MLP(in_dim,hidden_dim,inner_dim,2)
        # self.align_x=MLP(in_dim,hidden_dim,inner_dim,2)
        
        self.refine_cond=nn.ModuleList([
            nn.ModuleList([
                Trans_block(1,8,128,128,inner_dim,hidden_dim,0.1), # condition reps attn rel_ptoto    
                Trans_block(1,8,128,128,inner_dim,hidden_dim,0.1), # condition query attn condation reps
                Trans_block(1,8,128,128,inner_dim,hidden_dim,0.1), # condition query attn prior reps
            ]) for _ in range(3)
        ])
        
        self.enhance_reps=Trans_block(3,8,128,128,inner_dim,hidden_dim,0.1)

        self.cond_gate=make_fc(inner_dim,inner_dim,use_gn=True)
        self.cond_bias=make_fc(2*inner_dim,inner_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x, beta, context, condition_reps,rel_proto,t,rel_nums):
        time_emb=self.time_embedding(torch.tensor(t,device=context.device))
        
        condition_reps,context=self.align_cond(condition_reps),self.align_ctx(context)
        cond_query=self.cond_query.unsqueeze(0).expand(sum(rel_nums),-1)
        for cond_proto,query_cond,query_ctx in self.refine_cond:
            condition_reps=cond_proto(condition_reps,rel_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,rel_proto.shape[0])
            
            cond_query=query_cond(cond_query,condition_reps,rel_nums)
            cond_query=query_ctx(cond_query,context,rel_nums)
        
        x=x+time_emb
        enhanced_x=self.enhance_reps(x,cond_query,rel_nums)
        
        enhanced_x=enhanced_x*self.cond_gate(cond_query)+self.cond_bias(torch.cat([time_emb,cond_query],dim=-1))
        
        return enhanced_x
    

class FDM_Base(nn.Module):
    def __init__(self,cfg,in_dim,hidden_dim,cluster_num,cof=0.25):
        super().__init__()
        self.cond_query=nn.Parameter(torch.normal(mean=0, std=0.1, size=(hidden_dim,)))
        
        self.align_union=MLP(cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM,hidden_dim,hidden_dim,2)
        self.align_ctx=MLP(in_dim,hidden_dim,hidden_dim,2)
        
        self.encoder=nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    Trans_block(1,8,128,128,hidden_dim,hidden_dim//2,0.1), # condition reps attn rel_ptoto    
                    Trans_block(1,8,128,128,hidden_dim,hidden_dim//2,0.1), # condition query attn condation reps
                    Trans_block(1,8,128,128,hidden_dim,hidden_dim//2,0.1), # condition query attn prior reps
                ]) for _ in range(3)
            ]),
            MLP(hidden_dim,hidden_dim//2,hidden_dim,2)
        ])
        
        self.q_embed=VectorDiscrete(cluster_num,hidden_dim,commitment_cost=cof)
        
        self.decoder=nn.ModuleList([
            MLP(hidden_dim,hidden_dim//2,hidden_dim,2),
            Trans_block(1,8,128,128,hidden_dim,hidden_dim,0.1),
            MLP(hidden_dim,hidden_dim//2,hidden_dim,2),
        ])
        self.decode_proto=nn.Sequential(
            MLP(in_dim,hidden_dim,hidden_dim,2),
            nn.Sigmoid(),
            make_fc(hidden_dim,hidden_dim)
        )
    
    def init_vq_embed(self,proto_weight):
        if not getattr(self,'finish_init_embed',False):
            self.q_embed.init_embed_weight(proto_weight)
            setattr(self,'finish_init_embed',True)
            print('using pretrained relation prototype to init quantizer embedding')
            
    def encode(self,context,rel_proto,union_reps,rel_nums):
        union_reps=self.align_union(union_reps)
        context,rel_proto=self.align_ctx(context),self.align_ctx(rel_proto)
        
        cond_query=self.cond_query.unsqueeze(0).expand(sum(rel_nums),-1)
        for cond_proto,query_cond,query_ctx in self.encoder[0]:
            union_reps=cond_proto(union_reps,rel_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,rel_proto.shape[0])
            
            cond_query=query_cond(cond_query,union_reps,rel_nums)
            cond_query=query_ctx(cond_query,context,rel_nums)
        
        cond_query=self.encoder[1](cond_query)
        return cond_query
        
    def decode(self,q_ctx,rel_proto,rel_nums):
        decode_proto=self.decode_proto(rel_proto)
        
        q_ctx=self.decoder[0](q_ctx)
        q_ctx=self.decoder[1](q_ctx,decode_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,decode_proto.shape[0])
        q_ctx=self.decoder[2](q_ctx)

        return q_ctx,decode_proto
    
    def forward(self,context,rel_proto,union_reps,rel_nums,rel_labels=None):
        self.init_vq_embed(rel_proto)
        
        encode_ctx=self.encode(context,rel_proto,union_reps,rel_nums)
        
        q_ctx,q_loss,q_indices=self.q_embed(encode_ctx)
        
        decode_ctx,decode_proto=self.decode(q_ctx,rel_proto,rel_nums)
        
        losses=dict(q_loss=q_loss)
        
        return self.losses(decode_ctx,decode_proto,losses,rel_labels)

    def losses(self,ctx,proto,losses=dict(),rel_labels=None):
        if isinstance(rel_labels,(list,tuple)):
            rel_labels=torch.cat(rel_labels,dim=0)
        proto_norm = proto / proto.norm(dim=1, keepdim=True)
        ctx_norm=ctx/ctx.norm(dim=1,keepdim=True)
        pre_dist=(ctx_norm @ proto_norm.t()).softmax(-1)
        
        if not self.training or rel_labels is None:
            return pre_dist,proto,dict()
        
        rel_cls_weight=getattr(self,'rel_cls_weight',None).to(device=ctx.device) if getattr(self,'rel_cls_weight',None) is not None else None
        losses['dist_loss']=losses.get('dist_loss',0.0)+F.cross_entropy(pre_dist,rel_labels,weight=rel_cls_weight)
        
        if getattr(self,'predicate_reps_loss',None) is not None:
            losses=self.predicate_reps_loss(ctx,proto,rel_labels,losses,'intra_cls_loss','ctx_proto_dist')

        return pre_dist,proto,losses

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

        
class VectorDiscrete(nn.Module):
    def __init__(self, cluster_num, embedding_dim, commitment_cost=0.25):

        super(VectorDiscrete, self).__init__()
        self.cluster_num = cluster_num
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(cluster_num, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / cluster_num, 1 / cluster_num)

    def init_embed_weight(self,weight):
        reduced_weight=weight.data[:,:self.embedding_dim].to(device=self.embedding.weight.device)
        self.embedding.weight.data.copy_(reduced_weight)
        self.embedding.weight.requires_grad=True
    
    def forward(self, z):
        distances = (
            torch.sum(z**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [n, 1]
        encodings = torch.zeros(encoding_indices.size(0), self.cluster_num, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view_as(z)  # embedding weight ==> z

        e_latent_loss = F.mse_loss(quantized.detach(), z)  #  z --> quantized
        q_latent_loss = F.mse_loss(quantized, z.detach())  #  quantized --> z
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z + (quantized - z).detach()

        return quantized, loss, encoding_indices

class FDM(FDM_Base):
    def __init__(self, cfg, in_dim, hidden_dim, cluster_num,pre_codebook=False, num_steps=50,cof=0.25):
        super().__init__(cfg, in_dim, hidden_dim, cluster_num, cof)

        self.num_steps=num_steps
        self.diff_module=FED(hidden_dim,num_steps=num_steps,pre_codebook=pre_codebook)
        self.pre_codebook=pre_codebook
        
        self.extra_embedding=nn.Parameter(torch.normal(mean=0, std=0.1, size=(num_steps+1,cluster_num,hidden_dim)))
        setattr(self.q_embed,'forward',self.vq_forward)

        setattr(self.diff_module,'q_embed',self.q_embed)
        setattr(self.diff_module,'predicate_reps_loss',self.predicate_reps_loss)
        
        self.decode_proto=nn.Sequential(
            make_fc(hidden_dim,hidden_dim),
            nn.Sigmoid(),
            make_fc(hidden_dim,hidden_dim)
        )
        
        self.init_mean = MLP(hidden_dim,hidden_dim//2,hidden_dim,2)
        self.init_std = MLP(hidden_dim,hidden_dim//2,hidden_dim,2)

    
    def vq_forward(self, z, t, rel_labels=None):
        
        t=torch.tensor(t,device=z.device) if t is not None and isinstance(t,(tuple,list)) else t
        
        if t is not None:
            q_embed_w=self.q_embed.embedding.weight.unsqueeze(0).expand(z.shape[0],-1,-1)
            step_embed_w=self.extra_embedding[t]
            vq_embed_weight=q_embed_w+step_embed_w

            z_expand=z.unsqueeze(1).expand(-1,self.q_embed.cluster_num,-1)
            distances=torch.sum((z_expand-vq_embed_weight)**2,dim=-1)
            encoding_indices = torch.argmin(distances, dim=1)  # (n, )
            quantized = vq_embed_weight[torch.arange(z.shape[0]),encoding_indices]  # embedding weight ==> z

        else:
            distances = (
                torch.sum(z**2, dim=1, keepdim=True) + torch.sum(self.q_embed.embedding.weight**2, dim=1) - 2 * torch.matmul(z, self.q_embed.embedding.weight.t())
            )
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [n, 1]
            encodings = torch.zeros(encoding_indices.size(0), self.q_embed.cluster_num, device=z.device)
            encodings.scatter_(1, encoding_indices, 1)

            quantized = torch.matmul(encodings, self.q_embed.embedding.weight).view_as(z)  # embedding weight ==> z

        e_latent_loss = F.mse_loss(quantized.detach(), z)  #  z --> quantized
        q_latent_loss = F.mse_loss(quantized, z.detach())  #  quantized --> z
        loss = q_latent_loss + self.q_embed.commitment_cost * e_latent_loss
        
        losses=dict(q_loss=loss)
        if rel_labels is not None:
            rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(tuple,list)) else rel_labels
            oh_labels=F.one_hot(rel_labels,self.q_embed.cluster_num)
            dis_loss=F.mse_loss(distances,oh_labels.float())
            losses['q_dis_loss']=losses.get('q_dis_loss',0.0)+dis_loss
            
        quantized = z + (quantized - z).detach()

        return quantized, losses, encoding_indices
    
    def forward(self,context,rel_proto,union_reps,rel_nums,rel_labels=None,flexibility=0.0, ret_traj=False):
        self.init_vq_embed(rel_proto)
        losses=dict()
        
        encode_ctx,encode_proto=self.encode(context,rel_proto,union_reps,rel_nums)
        
        # encode_proto=self.decode_proto(encode_proto)
        # if self.pre_codebook:
        #     encode_ctx,q_loss_dict,q_indices=self.q_embed(encode_ctx,None,rel_labels)
        #     losses.update(q_loss_dict)
        mean,std=self.init_mean(encode_ctx),self.init_std(encode_ctx)
        x_T=mean+std*torch.randn((sum(rel_nums),encode_ctx.shape[-1]),device=encode_ctx.device)
        if self.training:
            rel_labels= torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
            prior_reps=encode_proto[rel_labels]
        
            e_theta,e_rand=self.diff_module(prior_reps,context=encode_ctx,rel_proto=encode_proto,rel_nums=rel_nums,rel_labels=rel_labels)    # input relation reps and reparameter latent reps  

            recon_loss = F.mse_loss(e_theta.view(-1, encode_ctx.shape[-1]), e_rand.view(-1, encode_ctx.shape[-1]), reduction='mean')
            losses['diffusion_recon_loss']=losses.get('diffusion_recon_loss',0.0)+recon_loss

            # losses=self.init_proto_loss(encode_proto,encode_proto / encode_proto.norm(dim=1, keepdim=True),losses)

            recon_q_ctx,losses=self.diff_module.step_by_step_sample_training(x_T=x_T,x_0=prior_reps,context=encode_ctx,rel_proto=encode_proto,rel_nums=rel_nums,rel_labels=rel_labels,add_losses=losses)
            # recon_q_ctx,losses=self.diff_module.sample_for_training(x_T=x_T,x_0=prior_reps,context=encode_ctx,rel_proto=encode_proto,rel_nums=rel_nums,rel_labels=rel_labels,add_losses=losses)
            # recon_q_ctx=self.diff_module.sample(x_T=x_T,context=encode_ctx,rel_proto=encode_proto,rel_nums=rel_nums,flexibility=flexibility,ret_traj=False)
        else:
            with torch.no_grad():
                recon_q_ctx=self.diff_module.sample(x_T=x_T,context=encode_ctx,rel_proto=encode_proto,rel_nums=rel_nums,flexibility=flexibility,ret_traj=ret_traj)
        
        if self.pre_codebook:
            recon_q_ctx,q_loss_dict,q_indices=self.q_embed(recon_q_ctx,None,rel_labels)
            for q_name,q_val in q_loss_dict.items():
                losses[q_name]=losses.get(q_name,0.0)+q_val
                
        decode_ctx,decode_proto=self.decode(recon_q_ctx,encode_proto.detach(),rel_nums)
        
        return self.losses(decode_ctx,decode_proto,losses,rel_labels)

    # def decode(self,q_ctx,rel_proto,rel_nums):
    #     q_ctx=self.decoder[0](q_ctx)
    #     q_ctx=self.decoder[1](q_ctx,rel_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,rel_proto.shape[0])
    #     q_ctx=self.decoder[2](q_ctx)

    #     return q_ctx
    
    def encode(self,context,rel_proto,union_reps,rel_nums):
        union_reps=self.align_union(union_reps)
        context,rel_proto=self.align_ctx(context),self.align_ctx(rel_proto)
        
        cond_query=self.cond_query.unsqueeze(0).expand(sum(rel_nums),-1)
        for cond_proto,query_cond,query_ctx in self.encoder[0]:
            union_reps=cond_proto(union_reps,rel_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,rel_proto.shape[0])
            
            cond_query=query_cond(cond_query,union_reps,rel_nums)
            cond_query=query_ctx(cond_query,context,rel_nums)
        
        cond_query=self.encoder[1](cond_query)
        return cond_query,rel_proto
 
    
class FED(nn.Module):
    def __init__(self,in_dim,num_steps=50,out_dims=[128,256,512,256,128],residual=True,pre_codebook=False):
        super().__init__()
        # ********* init parameter *********
        self.num_steps=num_steps
        beta_1,beta_T,sched_mode=1e-4,0.02,'linear'
        self.in_dim=in_dim
        self.out_dims=out_dims
        self.residual=residual
        self.pre_codebook=pre_codebook
        
        self.net=FEDBlock(in_dim,num_steps,out_dims=self.out_dims,residual=self.residual)

        self.var_sched=VarianceSchedule(self.num_steps,beta_1,beta_T,mode=sched_mode)
        
    def forward(self, x_0, context, rel_proto=None,rel_nums=None, t=None,rel_labels=None):
        """
        Args:
            x_0:  Input proto representation, (B, d).
            context:  Shape latent, (B, F).
            condition_reps: condition rel representation, (B, d)
        """
        batch_size, reps_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        else:
            t=[t]*batch_size
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1)       # (B, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)   # (B, 1)

        e_rand = torch.randn_like(x_0)  # (B, d)

        e_theta = self.net(c0 * x_0 + c1 * e_rand,context=context, rel_proto=rel_proto,t=t,rel_nums=rel_nums)

        return e_theta,e_rand
    
    def step_by_step_sample_training(self,x_T,x_0, context, rel_proto=None,rel_nums=None,rel_labels=None,add_losses=None,sample_steps=10):
        batch_size = x_0.size(0)

        sample_steps=sorted(random.sample(range(1,self.num_steps+1),sample_steps),reverse=True)
        if 1 not in sample_steps:
            sample_steps.append(1)
        if self.num_steps not in sample_steps:
            sample_steps.insert(0,self.num_steps)
            
        x_next=x_T
        for t in sample_steps:
            
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_next, context=context, rel_proto=rel_proto, t=[t]*batch_size,rel_nums=rel_nums)
            x_next = c0 * (x_next - c1 * e_theta)
            if not self.pre_codebook:
                x_next,q_loss_dict,q_indices=self.q_embed(x_next,[t]*batch_size,rel_labels)
                for q_name,q_val in q_loss_dict.items():
                    add_losses[q_name]=add_losses.get(q_name,0.0)+q_val
                    
        if self.training:
            add_losses['recon_reps']=add_losses.get('recon_reps',0.0)+F.mse_loss(x_next,x_0)
                
        return x_next,add_losses
    
    def sample_for_training(self,x_T,x_0, context, rel_proto=None,rel_nums=None,rel_labels=None,add_losses=None,sample_steps=10):
        batch_size = x_0.size(0)
        
        def add_noise(x_0,t):
            t=[t]*batch_size
            alpha_bar = self.var_sched.alpha_bars[t]
            beta = self.var_sched.betas[t]

            c0 = torch.sqrt(alpha_bar).view(-1, 1)       # (B, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)   # (B, 1)

            e_rand = torch.randn_like(x_0)  # (B, d)
            
            return c0 * x_0 + c1 * e_rand,e_rand

        sample_steps=sorted(random.sample(range(1,self.num_steps+1),sample_steps),reverse=True)
        if 1 not in sample_steps:
            sample_steps.append(1)
        if self.num_steps not in sample_steps:
            sample_steps.insert(0,self.num_steps)
        for t in sample_steps:
            x_t,z=add_noise(x_0,t)
            if t==self.num_steps:
                x_t=x_T
            
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, context=context, rel_proto=rel_proto, t=[t]*batch_size,rel_nums=rel_nums)
            x_next = c0 * (x_t - c1 * e_theta)
            if not self.pre_codebook:
                x_next,q_loss_dict,q_indices=self.q_embed(x_next,[t]*batch_size,rel_labels)
                for q_name,q_val in q_loss_dict.items():
                    add_losses[q_name]=add_losses.get(q_name,0.0)+q_val
            if self.training:
                add_losses['recon_noise']=add_losses.get('recon_noise',0.0)+F.mse_loss(e_theta,z)
                add_losses['recon_reps']=add_losses.get('recon_reps',0.0)+F.mse_loss(x_next,x_0)
                # add_losses=self.predicate_reps_loss(x_next,rel_proto,rel_labels,add_losses,loss_fun='intra_cls_loss',loss_name='recon_dis_proto')
                # rel_prot_norm = rel_proto / rel_proto.norm(dim=1, keepdim=True)
                # recon_norm=x_next/x_next.norm(dim=1,keepdim=True)
                # tail_pre=(recon_norm @ rel_prot_norm.t() ).softmax(-1)
                # add_losses['recon_ce_loss']=add_losses.get('recon_ce_loss',0.0)+F.cross_entropy(tail_pre,rel_labels)
                # add_losses=self.predicate_reps_loss(x_next,x_0,rel_labels,add_losses,loss_fun='inter_cls_loss',loss_name='recon_reps_dist')
                # add_losses=self.contrast_loss(x_next,x_0,add_losses)
        return x_next,add_losses

    def sample(self,x_T, context, rel_proto=None,rel_nums=None,flexibility=0.0, ret_traj=False):
        batch_size = x_T.size(0)
        x_T = x_T.to(x_T.device)
        traj = {self.num_steps: x_T}
        for t in range(self.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]

            e_theta = self.net(x_t, context=context, rel_proto=rel_proto, t=[t]*batch_size,rel_nums=rel_nums)
            x_next = c0 * (x_t - c1 * e_theta)
            if not self.pre_codebook:
                x_next,_,_=self.q_embed(x_next,[t]*batch_size)
            # x_next=x_next+ sigma * z
            traj[t-1] = x_next
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if ret_traj:
            traj[0] = traj[0].cpu()
            return traj
        else:
            return traj[0]
    
    def contrast_loss(self,reps1,reps2,losses=dict(),temperature=0.07):
        reps1_norm=F.normalize(reps1,dim=1)
        reps2_norm=F.normalize(reps2,dim=1)

        similarity_matrix = torch.matmul(reps1_norm, reps2_norm.T) 
        
        # expand_reps1,expand_reps2=reps1.unsqueeze(1),reps2.unsqueeze(0)
        # manh_dis=torch.sqrt(torch.sum((expand_reps1 - expand_reps2) ** 2, dim=2))

        mask = torch.eye(reps1.shape[0], device=reps1.device).bool()

        similarity_matrix /= temperature
        pos_sim=similarity_matrix[mask].view(reps1.shape[0], -1)
        neg_sim=similarity_matrix[~mask].view(reps1.shape[0], -1)
        sim_loss = -torch.log(torch.exp(pos_sim) / torch.exp(neg_sim).sum(dim=1))
        
        losses['sim_loss']=losses.get('sim_loss',0.0)+sim_loss.mean()

        # manh_dis /= temperature
        # pos_dis=manh_dis[mask].view(reps1.shape[0], -1)
        # neg_dis=manh_dis[~mask].view(reps1.shape[0], -1)
        # dis_loss = -torch.log(torch.exp(pos_dis) / torch.exp(neg_dis).sum(dim=1))

        # losses['dis_loss']=losses.get('dis_loss',0.0)+dis_loss.mean()

        return losses

class FEDBlock(nn.Module):
    def __init__(self,in_dim,num_steps,out_dims=[128,256,512,256,128],residual=True) -> None:
        super().__init__()
        
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            ConcatSquashLinear(in_dim if idx==0 else out_dims[idx-1], out_dim, in_dim) for idx,out_dim in enumerate(out_dims)
        ])
        self.layers.append(ConcatSquashLinear(out_dims[-1],in_dim,in_dim))
        
        self.time_embedding=nn.Embedding(num_steps+1,in_dim)
        nn.init.normal_(self.time_embedding.weight, mean=0, std=1)
        
        self.filter_time=Trans_block(1,8,128,128,in_dim,in_dim//2)
        self.enhance_ctx=Trans_block(1,8,128,128,in_dim,in_dim//2)
        self.enhance_proto=Trans_block(1,8,128,128,in_dim,in_dim//2)

    def forward(self, lt_reps, context, rel_proto, rel_nums,**kwargs):
        """
        Args:
            lt_reps:  latent representations, shape: (N,d).

            condition_reps: condition reps, (B, 3d)
        """
        
        time_emb=self.time_embedding(torch.tensor(kwargs['t'],device=context.device))
        lt_reps=self.filter_time(lt_reps,time_emb,rel_nums)
        
        lt_reps=self.enhance_ctx(lt_reps,context,rel_nums)
        lt_reps=self.enhance_proto(lt_reps,rel_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums)
        
        out = lt_reps
        for i, layer in enumerate(self.layers):
            out = layer(ctx=context,x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return lt_reps + out
        else:
            return out
