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
        enc1, enc2, enc3, att_weight_1, att_weight_2, att_weight_3 = self.transformer(emb1, emb2, emb3)
        r1, r2, r3 = self.rotatory_attention(emb1, emb2, emb3)
        
        att_weights = []
        rot_weights = []
        if self.vis:
            att_weights = [att_weight_1, att_weight_2, att_weight_3]
            rot_weights = [r1, r2, r3]

        # combine both  intra-slice and inter-slice information
        f1 = enc1 + r1
        f2 = enc2 + r2
        f3 = enc3 + r3
        
        o1, o2, o3 = self.reconstruct(f1, f2, f3)
        y = self.decoder(o1, o2, o3, x4)
        return self.out(y), att_weights, rot_weights
        
if __name__ == '__main__':
    import torch
    from config import get_config
    input = torch.rand(3, 1, 128, 128).cuda()
    model = RotCAtt_TransUNet_plusplus(get_config()).cuda()
    output, att_weights, rot_weights = model(input)
    print(output.size(), len(att_weights), len(rot_weights))
    print(len(att_weights[0]), len(att_weights[1]),  len(att_weights[2]))
    print(rot_weights[0].size(), rot_weights[1].size(), rot_weights[2].size())
    