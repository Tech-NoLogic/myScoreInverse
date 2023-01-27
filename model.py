from torch import nn

class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()