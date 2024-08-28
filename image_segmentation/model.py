import torch
from torch import nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.down = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(),
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.up = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.ReLU(),
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        self.down1 = DownBlock(64, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 128)
        self.down4 = DownBlock(128, 256)

        self.up4 = UpBlock(256, 128)
        self.up3 = UpBlock(128 * 2, 128)
        self.up2 = UpBlock(128 * 2, 64)
        self.up1 = UpBlock(64 * 2, 64)

        self.conv_out = nn.Conv2d(64 * 2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.conv_in(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.up4(x4)
        x5 = torch.cat((x5, x3), 1)
        x6 = self.up3(x5)
        x6 = torch.cat((x6, x2), 1)
        x7 = self.up2(x6)
        x7 = torch.cat((x7, x1), 1)
        x8 = self.up1(x7)
        x8 = F.relu(torch.cat((x8, x0), 1))

        return self.conv_out(x8)
