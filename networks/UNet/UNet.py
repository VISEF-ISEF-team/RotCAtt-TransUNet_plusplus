import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.conv = ConvBlock(in_features, out_features)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_features + out_features, out_features)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Encoder
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # Bottleneck 
        self.b = ConvBlock(512, 1024)

        # Decoder 
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottle neck
        b = self.b(p4)

        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        output = self.outputs(d4)
        return output


