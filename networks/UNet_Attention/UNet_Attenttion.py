import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention_gate = AttentionGate(in_channels, out_channels)
        self.c1 = ConvBlock(in_channels[0] + out_channels, out_channels)

    def forward(self, x, s):
        x = self.up(x)
        s = self.attention_gate(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class UNet_Attention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        filters = [64, 128, 256, 512]
        
        self.e1 = EncoderBlock(1, filters[0])
        self.e2 = EncoderBlock(filters[0], filters[1])
        self.e3 = EncoderBlock(filters[1], filters[2])
        
        self.b1 = ConvBlock(filters[2], filters[3])

        self.d1 = DecoderBlock([filters[3], filters[2]], 256)
        self.d2 = DecoderBlock([filters[2], filters[1]], 128)
        self.d3 = DecoderBlock([filters[1], filters[0]], 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        # Bottle neck
        b1 = self.b1(p3)

        # Decoder
        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return output