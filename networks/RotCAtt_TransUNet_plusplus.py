from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from .dense_feature_extraction import Dense
from .linear_embedding import LinearEmbedding
from .transformer import Transformer
from .rotatory_attention import RotatoryAttention
from .recon import Reconstruction
from .uct_decoder import UCTDecoder

class RotCAtt_TransUNet_plusplus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vis = config.vis
        self.vis = True
        self.dense = Dense(config)
        self.linear_embedding = LinearEmbedding(config)
        self.transformer = Transformer(config)
        self.rotatory_attention = RotatoryAttention(config)
        self.reconstruct = Reconstruction(config)
        self.decoder = UCTDecoder(config)
        self.out = nn.Conv2d(config.df[0], config.num_classes, kernel_size=(1,1), stride=(1,1))
        
    def forward(self, x):
        x1, x2, x3, x4 = self.dense(x)
        emb1, emb2, emb3 = self.linear_embedding(x1, x2, x3)
        e1, e2, e3, a1_weights, a2_weights, a3_weights, c1_weights, c2_weights, c3_weights = self.transformer(emb1, emb2, emb3)
        r1, r2, r3 = self.rotatory_attention(emb1, emb2, emb3)
        
        att_weights = []
        rot_weights = []
        if self.vis:
            att_weights = [a1_weights, a2_weights, a3_weights]
            rot_weights = [r1, r2, r3]
            context_weights = [c1_weights, c2_weights, c3_weights]

        # combine both  intra-slice and inter-slice information
        f1 = e1 + r1
        f2 = e2 + r2
        f3 = e3 + r3
        
        o1, o2, o3 = self.reconstruct(f1, f2, f3)
        y = self.decoder(o1, o2, o3, x4)
        return self.out(y), att_weights, rot_weights, context_weights
        
