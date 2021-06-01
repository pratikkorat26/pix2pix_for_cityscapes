import torch
import torch.nn as nn
from utils import ConvBlock

#C64-C128-C256-C512
#286 × 286 discriminator:
#C64-C128-C256-C512-C512-C512

class Discriminator(nn.Module):
    def __init__(self , in_channels , features = [64 , 128 , 256 , 512]):
        super(Discriminator, self).__init__()
        self.features = features
        self.layers = []

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, self.features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2))

        in_channels = self.features[0]
        for feature in self.features[1:]:
            self.layers.append(
            ConvBlock(in_channels=in_channels,
                      out_channels=feature,
                      kernel_size=4,
                      stride=1 if feature == self.features[-1] else 2,
                      padding=1,
                      if_down=True,
                      use_dropout=False,
                      bias=False,
                      padding_mode="reflect"))
            in_channels=feature

        self.model = nn.Sequential(*self.layers)
        self.last = nn.Conv2d(in_channels, 1, kernel_size=5, stride=1, padding=0)
        self.last_acti = nn.Sigmoid()

    def forward(self , x,y):
        x = torch.cat([x,y],dim=1)
        x = self.initial(x)
        x = self.model(x)
        x = self.last(x)
        return self.last_acti(x)



if __name__ == '__main__':
    inp = torch.rand(size=(1 ,3 , 286 , 286))
    model = Discriminator(3)
    print(model)
    out = model(inp , inp)
    print(out.shape)