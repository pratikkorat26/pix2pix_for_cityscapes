import torch.nn as nn
import torch
from torchvision.utils import save_image



class ConvBlock(nn.Module):
    def __init__(self , in_channels, out_channels,use_dropout=False,use_batchnorm=True,if_down=True, which_act=["relu", "leaky"], **kwargs):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.if_down = if_down
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        if self.if_down:
            self.activation = which_act[1]
        else:
            self.activation = which_act[0]

        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels= out_channels,
                      **kwargs) if self.activation == "leaky"
            else nn.ConvTranspose2d(in_channels = in_channels, out_channels=out_channels , **kwargs),
            nn.BatchNorm2d(num_features=out_channels) if self.use_batchnorm else nn.Identity(),
            nn.Dropout(p=0.5) if not self.use_dropout else nn.Identity(),
            nn.LeakyReLU(negative_slope = 0.2) if self.if_down else nn.ReLU()
        )
    def forward(self , inputs):
        return self.block(inputs)


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to("cuda"), y.to("cuda")
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
