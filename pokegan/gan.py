from pokegan import Discriminator, Generator, batch_size
from pokegan.data import PokemonImageDataset, ImageProcessor
import torchvision.transforms as transforms
import torch

class GAN(object):
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
    
    def train(self, epochs=10):
        data_set = PokemonImageDataset(transform=transforms.Compose([
            ImageProcessor(),
            transforms.ToTensor()
        ]))
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=16, shuffle=True)
        for epoch in range(epochs):
            return