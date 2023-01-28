from torch import nn

import blocks

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.in_conv = blocks.InConv(in_channels, 64)

        self.down1 = blocks.Down(64, 128)
        self.down2 = blocks.Down(128, 256)
        self.down3 = blocks.Down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = blocks.Down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)

        self.up1 = blocks.Up(1024, 512)
        self.up2 = blocks.Up(512, 256)
        self.up3 = blocks.Up(256, 128)
        self.up4 = blocks.Up(128, 64)

        self.out_conv = blocks.OutConv(64, out_channels)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = self.drop3(x3)
        x4 = self.down4(x3)
        x4 = self.drop4(x4)

        x = self.up1(x3, x4)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        x = self.out_conv(x)

        return x

