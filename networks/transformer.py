import torch
import torch.nn as nn
import copy
import math

class MultiheadAttention(nn.Module):
    def __init__(self, config, level):
        super().__init__()
        self.vis = config.vis
        self.n_heads = config.num_heads
        self.df = config.df[level]
        self.dk = config.dk[level]
        self.dq = config.dq[level]
        self.dv = config.dv[level]
        self.dh = int(self.df / self.n_heads)
        
        self.W_Q = nn.Linear(self.df, self.dq)
        self.W_K = nn.Linear(self.df, self.dk)
        self.W_V = nn.Linear(self.df, self.dv)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(config.dropout_rate)
        self.proj_dropout = nn.Dropout(config.dropout_rate)
        
    def _decompose(self, x):
        new_shape = x.size()[:-1] + (self.n_heads, self.dh)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def _compose(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.df, )
        return x.view(*new_shape)
    
    def forward(self, x):
        Q = self._decompose(self.W_Q(x))
        K = self._decompose(self.W_K(x))
        V = self._decompose(self.W_V(x))
    
        S = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dh)
        A = self.softmax(S)
        att_weights = A if self.vis else None
        A = self.attn_dropout(A)

        C = self._compose(torch.matmul(A, V))
        C = self.proj_dropout(C)
        return C, att_weights
        
    
class MLP(nn.Module):
    
    def __init__(self,config, in_channel, mlp_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()  # F.gelu
        self.dropout = nn.Dropout(config.dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_ratio = config.mlp_ratio
        df = config.df
        self.attn_norm1 = nn.LayerNorm(df[0], eps=1e-6)
        self.attn_norm2 = nn.LayerNorm(df[1], eps=1e-6)
        self.attn_norm3 = nn.LayerNorm(df[2], eps=1e-6)
        
        self.multihead_attention1 = MultiheadAttention(config, level=0)
        self.multihead_attention2 = MultiheadAttention(config, level=1)
        self.multihead_attention3 = MultiheadAttention(config, level=2)
        
        self.ffn_norm1 = nn.LayerNorm(df[0], eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(df[1], eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(df[2], eps=1e-6)
        
        self.ffn1 = MLP(config, df[0], df[0]*mlp_ratio)
        self.ffn2 = MLP(config, df[1], df[1]*mlp_ratio)
        self.ffn3 = MLP(config, df[2], df[2]*mlp_ratio)
        
    def forward(self, z1, z2, z3):
        '''  Block 1 ''' 
        h1, h2, h3 = z1, z2, z3

        # Layer norm
        z1 = self.attn_norm1(z1)
        z2 = self.attn_norm2(z2)
        z3 = self.attn_norm3(z3)      
        
        # Multihead attention + residual
        c1, a1 = self.multihead_attention1(z1)
        c2, a2 = self.multihead_attention2(z2)
        c3, a3 = self.multihead_attention3(z3)
        
        z1 = c1 + h1
        z2 = c2 + h2
        z3 = c3 + h3
        
        '''  Block 2 '''
        h1, h2, h3 = z1, z2, z3
        
        # Layer norm + MLP + residual
        z1 = self.ffn1(self.ffn_norm1(z1)) + h1
        z2 = self.ffn2(self.ffn_norm2(z2)) + h2
        z3 = self.ffn3(self.ffn_norm3(z3)) + h3

        return z1, z2, z3, a1, a2, a3, c1, c2, c3
        

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        df = config.df
        self.vis = config.vis
        self.layers = nn.ModuleList()
        self.encoder_norm1 = nn.LayerNorm(df[0], eps=1e-6)
        self.encoder_norm2 = nn.LayerNorm(df[1], eps=1e-6)
        self.encoder_norm3 = nn.LayerNorm(df[2], eps=1e-6)
        
        for _ in range(config.num_layers):
            layer = TransLayer(config)
            self.layers.append(copy.deepcopy(layer))
        
    def forward(self, z1, z2, z3):
        a1_weights = []
        a2_weights = []
        a3_weights = []
        c1_weights = []
        c2_weights = []
        c3_weights = []
        z1_list = [z1]
        z2_list = [z2]
        z3_list = [z3]
        
        for layer_block in self.layers:
            z1, z2, z3, a1, a2, a3, c1, c2, c3 = layer_block(z1, z2, z3)
            z1_list.append(z1)
            z2_list.append(z2)
            z3_list.append(z3)
            if self.vis: 
                a1_weights.append(a1)
                a2_weights.append(a2)
                a3_weights.append(a3)
                c1_weights.append(c1)
                c2_weights.append(c2)
                c3_weights.append(c3)
            
        e1 = self.encoder_norm1(z1)
        e2 = self.encoder_norm2(z2) 
        e3 = self.encoder_norm3(z3) 
        
        return e1, e2, e3, z1_list, z2_list, z3_list, a1_weights, a2_weights, a3_weights, c1_weights, c2_weights, c3_weights