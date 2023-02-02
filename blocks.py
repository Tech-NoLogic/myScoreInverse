import torch
from torch import nn
from torch.nn import functional as F
import math
import einops

class TimeEmbedding(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.dim = dim # 28
        self.time_dim = dim * 4 # 28 * 4 = 112
        self.time_embedding = nn.Sequential(
            nn.Linear(dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

    def forward(self, t):
        # [b,] -> [b, dim]
        
        half_dim = self.dim // 2 
        # dim / 2
        emb = math.log(10000) / (half_dim - 1) 
        # log(10000) / (dim/2 - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb) 
        # [1,2,...,dim/2]*-log(10000)/(dim/2-1)
        emb = t[:, None] * emb[None, :] # [b, 1]*[1, dim/2]->[b, dim/2]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # [b, dim]

        # [b, dim] -> [b, time_dim]
        return self.time_embedding(emb)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResidualDoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, mid_channels=None):
        super().__init__()
        if mid_channels == None:
            mid_channels = out_channels

        self.conv1 = ConvBlock(in_channels, mid_channels)
        self.conv2 = ConvBlock(mid_channels, out_channels)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, mid_channels)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = einops.rearrange(self.mlp(time_emb), "b c -> b c 1 1") + h
        h = self.conv2(h) + self.res_conv(x)

        return h


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim) -> None:
        super().__init__()

        self.in_conv = ResidualDoubleConvBlock(in_channels, out_channels, time_dim)
    
    def forward(self, x, time_emb):
        return self.in_conv(x, time_emb)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim) -> None:
        super().__init__()

        self.down = nn.ModuleList([
            nn.MaxPool2d(2),
            ResidualDoubleConvBlock(in_channels, out_channels, time_dim)
        ])

    # def forward(self, x, time_emb):
    #     return self.down(x, time_emb)
    # def forward(self, input):
    #     return self.down(input[0], input[1])
    def forward(self, x, time_emb):
        x = self.down[0](x)
        x = self.down[1](x, time_emb)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, bilinear=True) -> None:
        super().__init__()

        if bilinear:
            self.up = self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
                )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.double_conv = ResidualDoubleConvBlock(in_channels, out_channels, time_dim)

    def forward(self, x_down, x_left, time_emb):
        x_down = self.up(x_down)

        diff_x = x_left.shape[2] - x_down.shape[2]
        diff_y = x_left.shape[3] - x_down.shape[3]

        x_down = F.pad(x_down, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

        x = torch.cat((x_down, x_left), dim=1) # in_channels // 2 * 2 -> in_channels

        return self.double_conv(x, time_emb)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.out_conv(x)

