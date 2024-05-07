import torch
from torchinfo import summary

# Network 1: RotCAtt_TransUNet_plusplus
from networks.RotCAtt_TransUNet_plusplus.RotCAtt_TransUNet_plusplus import RotCAtt_TransUNet_plusplus
from networks.RotCAtt_TransUNet_plusplus.config import get_config as rot_config

# Network 2: TransUNet
from networks.TransUNet.TransUNet import TransUNet
from networks.TransUNet.vit_configs import VIT_CONFIGS

# Network 3: UNet_plusplus
from networks.UNet_plusplus.UNet_plusplus import UNet_plusplus

# Network 4: UNet
from networks.UNet.UNet import UNet

# Network 5: ResUNet
from networks.ResUNet.ResUNet import ResUNet

# Network 6: UNet_Attention 
from networks.UNet_Attention.UNet_Attenttion import UNet_Attention

# Network 7: UNet_plusplus_Attention
from networks.UNet_plusplus_Attention.UNet_plusplus_Attention import UNet_plusplus_Attention

# Network 8: SwinUnet
from networks.SwinUNet.SwinUNet import SwinUNet
from networks.SwinUNet.config import sample_config as swin_config

# Network 9: SwinUNet Attention
from networks.SwinUNet_Attention.SwinUNet_Attention import SwinUNet_Attention
from networks.SwinUNet_Attention.SwinUNet_Attention import get_swin_unet_attention_configs


def check_params():
    input = torch.rand(6, 1, 512, 512)
    model_config = rot_config()
    model_config.img_size = 512
    model_config.num_layers = 9
    model = RotCAtt_TransUNet_plusplus(model_config)
    print(summary(model, input.shape))
    
    
check_params()