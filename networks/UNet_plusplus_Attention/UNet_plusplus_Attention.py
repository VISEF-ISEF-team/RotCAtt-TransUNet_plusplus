import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.W_d = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, e, d):
        out = self.relu(self.W_e(e) + self.W_d(d))
        out = self.output(out)
        return out * e

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
   
    
class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x
    
class Dense(nn.Module):
    def __init__(self, in_channels, out_channels, layer):
        super().__init__()
        self.layer = layer
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attention_gate = AttentionGate([in_channels[0], in_channels[1]], out_channels)
        self.conv = ConvBlock(in_channels[0]*layer+in_channels[1], out_channels)
        
    def forward(self, d, e1=None, e2=None, e3=None, e4=None):
        d = self.up(d)
        if self.layer==1: 
            e1 = self.attention_gate(e1, d)
            f = torch.cat([e1,d], dim=1)
        elif self.layer==2:
            e2 = self.attention_gate(e2, d)
            f = torch.cat([e1,e2,d], dim=1)
        elif self.layer==3: 
            e3 = self.attention_gate(e3, d)
            f = torch.cat([e1,e2,e3,d], dim=1)
        elif self.layer==4: 
            e4 = self.attention_gate(e4, d)
            f = torch.cat([e1,e2,e3,e4,d], dim=1)
            
        o = self.conv(f)
        return o


class UNet_plusplus_Attention(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super().__init__()
        filters = [1, 32, 64, 128, 256, 512]
        self.down1 = ConvBlock(filters[0], filters[1])
        
        # Downsampling
        self.down2 = Downsampling(filters[1], filters[2])
        self.down3 = Downsampling(filters[2], filters[3])
        self.down4 = Downsampling(filters[3], filters[4])
        self.down5 = Downsampling(filters[4], filters[5])

        self.dense1_2 = Dense([filters[1], filters[2]], filters[1], layer=1)
        self.dense2_2 = Dense([filters[2], filters[3]], filters[2], layer=1)
        self.dense3_2 = Dense([filters[3], filters[4]], filters[3], layer=1)
        self.dense4_2 = Dense([filters[4], filters[5]], filters[4], layer=1) 
        
        self.dense1_3 = Dense([filters[1], filters[2]], filters[1], layer=2)
        self.dense2_3 = Dense([filters[2], filters[3]], filters[2], layer=2)
        self.dense3_3 = Dense([filters[3], filters[4]], filters[3], layer=2)
        
        self.dense1_4 = Dense([filters[1], filters[2]], filters[1], layer=3)
        self.dense2_4 = Dense([filters[2], filters[3]], filters[2], layer=3)
        
        self.dense1_5 = Dense([filters[1], filters[2]], filters[1], layer=4)
        
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[1], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[1], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[1], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[1], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[1], num_classes, kernel_size=1)
    
    def forward(self, input):
        x1_1 = self.down1(input)
        
        x2_1 = self.down2(x1_1)
        x1_2 = self.dense1_2(x2_1, x1_1)
        
        x3_1 = self.down3(x2_1)
        x2_2 = self.dense2_2(x3_1, x2_1)
        x1_3 = self.dense1_3(x2_2, x1_1, x1_2)
        
        x4_1 = self.down4(x3_1)
        x3_2 = self.dense3_2(x4_1, x3_1)
        x2_3 = self.dense2_3(x3_2, x2_1, x2_2)
        x1_4 = self.dense1_4(x2_3, x1_1, x1_2, x1_3)
        
        x5_1 = self.down5(x4_1)
        x4_2 = self.dense4_2(x5_1, x4_1)
        x3_3 = self.dense3_3(x4_2, x3_1, x3_2)
        x2_4 = self.dense2_4(x3_3, x2_1, x2_2, x2_3)
        x1_5 = self.dense1_5(x2_4, x1_1, x1_2, x1_3, x1_4)
                
        if self.deep_supervision:            
            output1 = self.final1(x1_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x1_4)
            output4 = self.final4(x1_5)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x1_5)
            return output