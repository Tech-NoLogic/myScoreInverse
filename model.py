import torch
from torch import nn

import blocks
from config import Config

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim) -> None:
        super().__init__()
        self.time_embedding = blocks.TimeEmbedding(Config.img_size)
        self.in_conv = blocks.InConv(in_channels, 64, Config.time_dim)

        self.down1 = blocks.Down(64, 128, Config.time_dim)
        self.down2 = blocks.Down(128, 256, Config.time_dim)
        self.down3 = blocks.Down(256, 512, Config.time_dim)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = blocks.Down(512, 1024, Config.time_dim)
        self.drop4 = nn.Dropout2d(0.5)

        self.up1 = blocks.Up(1024, 512, Config.time_dim)
        self.up2 = blocks.Up(512, 256, Config.time_dim)
        self.up3 = blocks.Up(256, 128, Config.time_dim)
        self.up4 = blocks.Up(128, 64, Config.time_dim)

        self.out_conv = blocks.OutConv(64, out_channels)

    def forward(self, x, time):
        time_emb = self.time_embedding(time)

        x0 = self.in_conv(x, time_emb)

        x1 = self.down1(x0, time_emb)
        x2 = self.down2(x1, time_emb)
        x3 = self.down3(x2, time_emb)
        x3 = self.drop3(x3)
        x4 = self.down4(x3, time_emb)
        x4 = self.drop4(x4)

        x = self.up1(x4, x3, time_emb)
        x = self.up2(x, x2, time_emb)
        x = self.up3(x, x1, time_emb)
        x = self.up4(x, x0, time_emb)

        x = self.out_conv(x)

        return x

# in_channels = 3
# out_channels = 3
# time_dim = 28 * 4
# u = UNet(in_channels, out_channels, time_dim)

# x = torch.randn(4, 3, 28, 28)
# time = torch.randint(0, 200, (4,))
# y = u(x, time)
