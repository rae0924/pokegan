from pokegan import image_size, num_channels
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 3 * 100 * 100 -> 100 * 50 * 50
            nn.Conv2d(in_channels=num_channels, out_channels=image_size,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # 100 * 50 * 50 -> 200 * 25 * 25
            nn.Conv2d(in_channels=image_size, out_channels=image_size*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=image_size*2),
            nn.LeakyReLU(negative_slope=0.1),
            # 200 * 25 * 25 -> 400 * 12 * 12
            nn.Conv2d(in_channels=image_size*2, out_channels=image_size*4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=image_size*4),
            nn.LeakyReLU(negative_slope=0.1),
            # 400 * 12 * 12 -> 800 * 6 * 6
            nn.Conv2d(in_channels=image_size*4, out_channels=image_size*8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=image_size*8),
            nn.LeakyReLU(negative_slope=0.1),
            # 800 * 6 * 6 -> 1 * 1 * 1
            nn.Conv2d(in_channels=image_size*8, out_channels=1,
                      kernel_size=6, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
