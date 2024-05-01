import torch
import torch.nn as nn

class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
    
class UNet_plusplus(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()

        filters = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Normal conv for downsampling for all blocks
        self.conv1_1 = ConvBatchRelu(input_channels, filters[0])
        self.conv2_1 = ConvBatchRelu(filters[0], filters[1])
        self.conv3_1 = ConvBatchRelu(filters[1], filters[2])
        self.conv4_1 = ConvBatchRelu(filters[2], filters[3])
        self.conv5_1 = ConvBatchRelu(filters[3], filters[4])

        # 2x layers for 4 blocks
        self.conv1_2 = ConvBatchRelu(filters[0] + filters[1], filters[0])
        self.conv2_2 = ConvBatchRelu(filters[1] + filters[2], filters[1])
        self.conv3_2 = ConvBatchRelu(filters[2] + filters[3], filters[2])
        self.conv4_2 = ConvBatchRelu(filters[3] + filters[4], filters[3])

        # 3 layers for 3 blocks
        self.conv1_3 = ConvBatchRelu(filters[0] * 2 + filters[1], filters[0])
        self.conv2_3 = ConvBatchRelu(filters[1] * 2 + filters[2], filters[1])
        self.conv3_3 = ConvBatchRelu(filters[2] * 2 + filters[3], filters[2])

        # 4 layers for 2 blocks
        self.conv1_4 = ConvBatchRelu(filters[0] * 3 + filters[1], filters[0])
        self.conv2_4 = ConvBatchRelu(filters[1] * 3 + filters[2], filters[1])

        # 5 layers for block 1
        self.conv1_5 = ConvBatchRelu(filters[0] * 4 + filters[1], filters[0])
        
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)


    def forward(self, input):
        x1_1 = self.conv1_1(input)
        
        x2_1 = self.conv2_1(self.pool(x1_1)) # downsampling
        x1_2 = self.conv1_2(torch.cat([x1_1, self.up(x2_1)], 1))  # block 2 -> block 1 = block 1 layer 2

        x3_1 = self.conv3_1(self.pool(x2_1)) # downsampling
        x2_2 = self.conv2_2(torch.cat([x2_1, self.up(x3_1)], 1))  # block 3 -> block 2 = block 2 layer 2
        x1_3 = self.conv1_3(torch.cat([x1_1, x1_2, self.up(x2_2)], 1)) # block 2 -> block 1 = block 1 layer 3

        x4_1 = self.conv4_1(self.pool(x3_1)) # downsampling
        x3_2 = self.conv3_2(torch.cat([x3_1, self.up(x4_1)], 1)) # block 4 -> block 3 = block 3 layer 2
        x2_3 = self.conv2_3(torch.cat([x2_1, x2_2, self.up(x3_2)], 1)) # block 3 -> block 2 = block 2 layer 3
        x1_4 = self.conv1_4(torch.cat([x1_1, x1_2, x1_3, self.up(x2_3)], 1)) # block 2 -> block 1 = block 1 layer 4

        x5_1 = self.conv5_1(self.pool(x4_1)) # downsampling
        x4_2 = self.conv4_2(torch.cat([x4_1, self.up(x5_1)], 1))  # block 5 -> block 4 = block 4 layer 2
        x3_3 = self.conv3_3(torch.cat([x3_1, x3_2, self.up(x4_2)], 1)) # block 4 -> block 3 = block 3 layer 3
        x2_4 = self.conv2_4(torch.cat([x2_1, x2_2, x2_3, self.up(x3_3)], 1)) # block 3 -> block 2 = block 2 layer 4
        x1_5 = self.conv1_5(torch.cat([x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))  # block 2 -> block 1 = block 1 layer 5
            
        if self.deep_supervision:            
            output1 = self.final1(x1_2)
            output2 = self.final2(x1_3)
            output3 = self.final3(x1_4)
            output4 = self.final4(x1_5)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x1_5)
            return output