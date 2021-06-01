from utils import ConvBlock
import torch
import torch.nn as nn

#C64-C128-C256-C512-C512-C512-C512-C512
class Generator(nn.Module):
    def __init__(self , in_channels ,feature, **kwargs):
        super(Generator, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, feature, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = ConvBlock(feature, out_channels=feature*2, kernel_size=4, stride=2, padding=1, if_down=True,
                               use_dropout=False , use_batchnorm = False)
        self.down2 = ConvBlock(feature*2, out_channels=feature * 4, kernel_size=4, stride=2, padding=1, if_down=True,
                               use_dropout=False)
        self.down3 = ConvBlock(feature*4,feature*8,kernel_size=4, stride=2, padding=1, if_down=True,
                               use_dropout=False)
        self.down4 = ConvBlock(feature * 8, out_channels=feature * 8, kernel_size=4, stride=2, padding=1, if_down=True,
                               use_dropout=False)
        self.down5 = ConvBlock(feature * 8, out_channels=feature * 8, kernel_size=4, stride=2, padding=1, if_down=True,
                               use_dropout=False)
        self.down6 = ConvBlock(feature * 8, out_channels=feature * 8, kernel_size=4, stride=2, padding=1, if_down=True,
                               use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature * 8, feature * 8, 4, 2, 1), nn.ReLU()
        )


        self.up1 = ConvBlock(feature*8,out_channels=feature*8,if_down=False,kernel_size=4,stride=2,padding=1,
                             use_dropout=True)
        self.up2 = ConvBlock(feature * 8 * 2, out_channels=feature * 8, if_down=False, kernel_size=4, stride=2, padding=1,
                             use_dropout=True)
        self.up3 = ConvBlock(feature * 8 * 2, out_channels=feature * 8, if_down=False, kernel_size=4, stride=2, padding=1,
                             use_dropout=True)
        self.up4 = ConvBlock(feature * 8 * 2, out_channels=feature * 8, if_down=False, kernel_size=4, stride=2, padding=1,
                             use_dropout=False)
        self.up5 = ConvBlock(feature * 8 * 2, out_channels=feature * 4, if_down=False, kernel_size=4, stride=2, padding=1,
                             use_dropout=False)
        self.up6 = ConvBlock(feature * 4 * 2, out_channels=feature * 2, if_down=False, kernel_size=4, stride=2, padding=1,
                             use_dropout=False)
        self.up7 = ConvBlock(feature * 2 * 2, out_channels=feature, if_down=False, kernel_size=4, stride=2, padding=1,
                             use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(feature * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def forward(self , x):
        d1 = self.initial_down(x)

        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        btlnck = self.bottleneck(d7)

        up1 = self.up1(btlnck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))

        x = self.final_up(torch.cat([up7,d1] , dim=1))
        return x



if __name__ == '__main__':
    model = Generator(in_channels=3,feature=64)
    print(model)
    inp = torch.rand(size=(1,3,256,256))
    out = model(inp)
    print(out.shape)