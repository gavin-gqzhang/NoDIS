import torch
import torch.nn as nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import MultiHeadAttention, PositionwiseFeedForward, EncoderLayer

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


class Single_Att_Layer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, att_wo_drop=False):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, att_wo_drop=att_wo_drop)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
    def forward(self, q_input, k_input, v_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q_input, k_input, v_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()
        
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()
        
        return enc_output, enc_slf_attn

class Attn_block(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1, att_wo_drop=False):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, att_wo_drop=att_wo_drop)
    
    def forward(self, q_feats, kv_feats=None, q_split=None, kv_len=None):
        """_summary_

        Args:
            q_feats (tensor): query representation
            kv_feats (tensor, default: None): key representation. Defaults to None.
            q_split (list, default: None): query tensor split numbers. Defaults to None.
            kv_split (list, default: None): key and value tensor split numbers. Defaults to None.

        Returns:
            (tensor, tensor, tensor, tensor): attention output, attention weight, pad mask, attention tensor after processed
        """
        # -- Prepare masks
        if q_split is not None:
            q_feats=q_feats.split(q_split,dim=0)
            q_feats = nn.utils.rnn.pad_sequence(q_feats, batch_first=True)
            
            bsz, device, pad_len = len(q_split), q_feats.device, max(q_split)
            
            q_lens = torch.LongTensor(q_split).to(device).unsqueeze(1).expand(-1, pad_len)
            non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(q_lens).unsqueeze(-1) # (bsz, pad_len, 1)
    
            if kv_len is not None:
                attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(q_lens).unsqueeze(1).expand(-1, kv_len, -1).transpose(1, 2)
            else:
                attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(q_lens).unsqueeze(1).expand(-1, pad_len, -1).clone() # (bsz, pad_len, pad_len)

                if kv_feats is not None:
                    kv_feats=kv_feats.split(q_split,dim=0)
                    kv_feats=nn.utils.rnn.pad_sequence(kv_feats, batch_first=True)
                else:
                    kv_feats=q_feats
           
            attn_mask = self.zero_check(q_split, attn_mask)
        else:
            non_pad_mask,attn_mask=None,None
            
        # -- Forward
        enc_output, enc_slf_attn = trans_layer(
                q_feats, kv_feats, kv_feats, 
                non_pad_mask=non_pad_mask, slf_attn_mask=attn_mask)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn, non_pad_mask, enc_output[non_pad_mask.squeeze(-1)]
    
    
    @staticmethod
    def zero_check(num_objs, slf_attn_mask):
        non_zero = [i for i, v in enumerate(num_objs) if v != 0]
        for i, v in enumerate(num_objs):
            if v == 0:
                nz_idx = non_zero[0]
                slf_attn_mask[i] = slf_attn_mask[nz_idx].clone()
        return slf_attn_mask


class Trans_block(nn.Module):
    def __init__(self,n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, att_wo_drop=False):
        super().__init__()
        self.trans_block = nn.ModuleList([
                Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, att_wo_drop=att_wo_drop)
            for _ in range(n_layers)])
        
    def forward(self, q_feats, kv_feats=None, q_split=None, kv_len=None):
        # -- Prepare masks
        if q_split is not None:
            q_feats=q_feats.split(q_split,dim=0)
            q_feats = nn.utils.rnn.pad_sequence(q_feats, batch_first=True)
            
            bsz, device, pad_len = len(q_split), q_feats.device, max(q_split)
            
            q_lens = torch.LongTensor(q_split).to(device).unsqueeze(1).expand(-1, pad_len)
            non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(q_lens).unsqueeze(-1) # (bsz, pad_len, 1)
    
            if kv_feats is not None and len(kv_feats.shape)==3:
                kv_len=kv_feats.shape[1]
                
            if kv_len is not None:
                attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(q_lens).unsqueeze(1).expand(-1, kv_len, -1).transpose(1, 2)
            else:
                attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(q_lens).unsqueeze(1).expand(-1, pad_len, -1).clone() # (bsz, pad_len, pad_len)

                if kv_feats is not None:
                    kv_feats=kv_feats.split(q_split,dim=0)
                    kv_feats=nn.utils.rnn.pad_sequence(kv_feats, batch_first=True)
                else:
                    kv_feats=q_feats
           
            attn_mask = self.zero_check(q_split, attn_mask)
        else:
            if kv_len is not None and len(kv_feats.shape)==2:
                bs,q_len,device,pad_len=q_feats.shape[:2],q_feats.device,max(kv_len)
                
                kv_feats=kv_feats.split(kv_len,dim=0)
                kv_feats = nn.utils.rnn.pad_sequence(kv_feats, batch_first=True)
            
                kv_pad_len=torch.LongTensor(kv_feats).to(device).unsqueeze(1).expand(-1, pad_len)
                
                attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bs, -1).ge(kv_pad_len).unsqueeze(1).expand(-1, q_len, -1)
                non_pad_mask=torch.ones((bs,q_len),device=device).unsqueeze(-1)
            else:
                non_pad_mask,attn_mask=None,None
            
        # -- Forward
        for trans_layer in self.trans_block:
            enc_output, enc_slf_attn = trans_layer(
                q_feats, kv_feats, kv_feats, 
                non_pad_mask=non_pad_mask, slf_attn_mask=attn_mask)
        enc_output = enc_output[non_pad_mask.squeeze(-1)]

        return enc_output
    
    @staticmethod
    def zero_check(num_objs, slf_attn_mask):
        non_zero = [i for i, v in enumerate(num_objs) if v != 0]
        for i, v in enumerate(num_objs):
            if v == 0:
                nz_idx = non_zero[0]
                slf_attn_mask[i] = slf_attn_mask[nz_idx].clone()
        return slf_attn_mask

