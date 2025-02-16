
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.nn.init as init
import numpy
import torch
from torch import nn 
import torch.nn.functional as F
import pandas as pd


import torch
import torch.nn as nn




class DynamicConvolution(nn.Module):

    def __init__(
        self,
        wshare,
        n_feat,
        dropout_rate,
        kernel_size,
        use_kernel_mask=False,
        use_bias=False,
    ):
        """Construct Dynamic Convolution layer."""
        super(DynamicConvolution, self).__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.attn = None

        # linear -> GLU -- -> lightconv -> linear
        #               \        /
        #                 Linear
        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        self.linear2 = nn.Linear(n_feat, n_feat)
        self.linear_weight = nn.Linear(n_feat, self.wshare * 1 * kernel_size)
        init.xavier_uniform_(self.linear_weight.weight)
        self.act = nn.GLU()

        # dynamic conv related
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))
            self.reset_parameters()

    def reset_parameters(self):

        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)
    def forward(self, query, key, value,count, mask):
        """Forward of 'Dynamic Convolution'.
        This function takes query, key and value but uses only quert.
        This is just for compatibility with self-attention layer (attention.py)
        Args:
            query (torch.Tensor): (batch, time1, d_model) input tensor
            key (torch.Tensor): (batch, time2, d_model) NOT USED
            value (torch.Tensor): (batch, time2, d_model) NOT USED
            mask (torch.Tensor): (batch, time1, time2) mask
        Return:
            x (torch.Tensor): (batch, time1, d_model) ouput
        """
        # linear -> GLU -- -> lightconv -> linear
        #               \        /
        #                 Linear
        x = query
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size
        
        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)
        # get kernel of convolution
        weight = self.linear_weight(x)  # B x T x kH
        weight = F.dropout(weight, self.dropout_rate, training=self.training)
        weight = weight.view(B, T, H, k).transpose(1, 2).contiguous()  # B x H x T x k
        weight_new = torch.zeros(B * H * T * (T + k - 1), dtype=weight.dtype)
        weight_new = weight_new.view(B, H, T, T + k - 1).fill_(float("-inf"))
        weight_new = weight_new.to(x.device)  # B x H x T x T+k-1
        weight_new.as_strided(
            (B, H, T, k), ((T + k - 1) * T * H, (T + k - 1) * T, T + k, 1)
        ).copy_(weight)
        weight_new = weight_new.narrow(-1, int((k - 1) / 2), T)  # B x H x T x T(k)
        if self.use_kernel_mask:
            kernel_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
            weight_new = weight_new.masked_fill(kernel_mask == 0.0, float("-inf"))
        weight_new = F.softmax(weight_new, dim=-1)
      
        self.attn = weight_new
        weight_new = weight_new.view(B * H, T, T)
        




        # convolution
        x = x.transpose(1, 2).contiguous()  # B x C x T
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)  # BH x T x C/H
        x = x.transpose(1, 2).contiguous().view(B, C, T)

        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)
        x = x.transpose(1, 2)  # B x T x C

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)

        # second linear layer
        x = self.linear2(x)
        return x




import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p,kernelsize,dropk,convdim):
        super().__init__()

        self.n_heads = n_heads 
        self.max_T = max_T
        self.drop_p=drop_p
        Dimratio=int(convdim)
        self.attdim=h_dim-Dimratio
        self.convdim= Dimratio
        if self.attdim !=0:
            self.q_net = nn.Linear(self.attdim, self.attdim)
            self.k_net = nn.Linear(self.attdim, self.attdim)
            self.v_net = nn.Linear(self.attdim, self.attdim)

            self.fmlp = nn.Sequential(
                nn.Linear(h_dim, h_dim//2),
                nn.ReLU(),
             )
            self.fl1=nn.Linear(h_dim//2, self.attdim)
            self.fl2=nn.Linear(h_dim//2, self.convdim)

        self.proj_net = nn.Linear(h_dim, h_dim)
        self.dropk=dropk
        self.att_drop = nn.Dropout(dropk)
        self.proj_drop = nn.Dropout(drop_p)
        
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones)
        self.kernelsize=kernelsize
    
        mask = mask.view(1, 1, max_T, max_T)
        
        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)
        self.Dynamicconv_layer = DynamicConvolution(
                                                wshare=self.convdim//4, 
                                                n_feat=self.convdim, 
                                                dropout_rate=0.2, 
                                                kernel_size= self.kernelsize, 
                                                use_kernel_mask=True, 
                                                use_bias=True
                                            ).to('cuda')
        self.count=0
    def forward(self, x):
        B, T, D = x.shape # batch size, seq length, h_dim * n_heads

        x1, x2 = torch.split(x,[self.attdim,self.convdim], dim=-1) #Separate input
        if self.attdim !=0:
            attention_future = torch.jit.fork(self.attention_branch, x1,B,T)
            conv_future = torch.jit.fork(self.conv_branch, x2)

            attention_output = torch.jit.wait(attention_future)
            conv_output = torch.jit.wait(conv_future)

            attention2= torch.cat((attention_output,conv_output),dim=-1)
            output=attention2
            Xbar=self.fmlp(attention2)

            X1=self.fl1(Xbar)
            X2=self.fl2(Xbar)
            Z= torch.cat((X1,X2),dim=-1)
            Xf=F.softmax(Z, dim=-1)
            output=Xf*attention2
           



        else:
            conv_future = torch.jit.fork(self.conv_branch, x2)
            output = torch.jit.wait(conv_future)
        
       

        self.count=self.count+1   #heatmap plot

        
        out = self.proj_drop(self.proj_net(output))
        
        #print(self.count)
        return out
    
    def attention_branch(self, x1,B,T):
        N, D = self.n_heads, x1.size(2) // self.n_heads

        q = self.q_net(x1).view(B, T, N, D).transpose(1,2) 
        k = self.k_net(x1).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x1).view(B, T, N, D).transpose(1,2)


        weights = q @ k.transpose(2, 3) / math.sqrt(D)
#       # Dropkey module
        # if self.training:
        #     mask_ratio=torch.ones_like(weights)*self.dropk
        #     weights= weights+torch.bernoulli(mask_ratio)*1e-13
        # attention = weights @ v

        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        
        attention = self.att_drop(weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        return attention
    
    def conv_branch(self, x2):

        # Dynamic Conv
        conv_output = self.Dynamicconv_layer(x2, x2, x2,self.count, mask=None)
        return conv_output


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p,kernelsize,dropk,convdim):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p,kernelsize,dropk,convdim)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
       
        return x





class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p,kernelsize,convdim, max_timestep=4096):
        super().__init__()


        self.act_dim = act_dim
        self.h_dim = h_dim

        input_seq_len = 3 * context_len


        blocks=[]
        dk=[0.1,0.1,0.1,0.1] #dropout rate of attention branch
        for i in range(n_blocks):
            dropk=dk[i]
            blocks.append(Block(h_dim, input_seq_len, n_heads, drop_p, kernelsize,dropk,convdim))
            
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )
        self.apply(self.init_weights)

    #Initialization 
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight)

    def forward(self, timesteps, states, actions, returns_to_go):

        B, T, _ = states.shape
       
        time_embeddings = self.embed_timestep(timesteps)

        returns_to_go = returns_to_go.float() 
        states=states.float() 
 
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings


        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)
       
        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3) #（B, 3, T, self.h_dim)

        # get predictions
        return_preds = self.predict_rtg(h[:,0])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s

        return state_preds, action_preds, return_preds