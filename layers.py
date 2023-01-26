from torch import nn

class UNet(nn.Module):
    def __init__(self, noise, timeStep) -> None:
        super().__init__()