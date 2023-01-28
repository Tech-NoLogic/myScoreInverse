import torch
from torch import nn
from torch.nn import functional as F

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super.__init__()
        if mid_channels == None:
            mid_channels = out_channels

        self.double_conv == nn.Sequential (
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.in_conv = DoubleConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        return self.in_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x_down, x_left):
        x_down = self.up(x_down)

        diff_x = x_left[2] - x_down[2]
        diff_y = x_left[3] - x_down[3]

        x_down = F.pad(x_down, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

        x = torch.cat(x_down, x_left, dim=1)

        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.out_conv(x)

