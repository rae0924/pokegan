from pokegan import image_size, num_channels, noise_size
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 100 * 1 * 1 -> 800 * 6 * 6
            nn.ConvTranspose2d(in_channels=noise_size, out_channels=image_size*8,
                      kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(num_features=image_size*8),
            nn.ReLU(),
            # 800 * 6 * 6 -> 400 * 12 * 12
            nn.ConvTranspose2d(in_channels=image_size*8, out_channels=image_size*4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=image_size*4),
            nn.ReLU(),
            # 400 * 12 * 12 -> 200 * 24 * 24
            nn.ConvTranspose2d(in_channels=image_size*4, out_channels=image_size*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=image_size*2),
            nn.ReLU(),
            # 200 * 24 * 24 -> 100 * 50 * 50
            nn.ConvTranspose2d(in_channels=image_size*2, out_channels=image_size,
                      kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(num_features=image_size),
            nn.ReLU(),
            # 100 * 50 * 50 -> 3 * 100 * 100
            nn.ConvTranspose2d(in_channels=image_size, out_channels=num_channels,
                      kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
