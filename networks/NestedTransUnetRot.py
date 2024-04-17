from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

# from config import get_config
from .dense_feature_extraction import Dense
from .linear_embedding import LinearEmbedding
from .transformer import Transformer
from .rotatory_attention import RotatoryAttention
from .recon import Reconstruction
from .decoder_cup import DecoderCup
from .uct_decoder import UCTDecoder

class NestedTransUnetRot(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = Dense(config)
        self.linear_embedding = LinearEmbedding(config)
        self.transformer = Transformer(config)
        self.rotatory_attention = RotatoryAttention(config)
        self.reconstruct = Reconstruction(config)
        if config.decoder == 'UCT': self.decoder = UCTDecoder(config)
        elif config.decoder == 'DecoderCup': self.decoder = DecoderCup(config)
        self.out = nn.Conv2d(config.df[0], config.num_classes, kernel_size=(1,1), stride=(1,1))
        
    def forward(self, x):
        x1, x2, x3, x4 = self.dense(x)
        emb1, emb2, emb3 = self.linear_embedding(x1, x2, x3)
        enc1, enc2, enc3 = self.transformer(emb1, emb2, emb3)
        r1, r2, r3 = self.rotatory_attention(emb1, emb2, emb3)

        # combine both  intra-slice and inter-slice information
        f1 = enc1 + r1
        f2 = enc2 + r2
        f3 = enc3 + r3
        
        if self.config.decoder == 'UCT':
            o1, o2, o3 = self.reconstruct(f1, f2, f3)
            self.decoder(o1, o2, o3, x4)
            y = self.decoder(o1, o2, o3, x4)
            return self.out(y)
        
# config = get_config()
# input = torch.rand(3, 1, 128, 128).cuda()
# model = NestedTransUnetRot(config).cuda()
# output = model(input)
# print(output.size())