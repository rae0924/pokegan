from pokegan import image_size, batch_size
from urllib.request import urlopen
from bs4 import BeautifulSoup
import torch.utils.data
import numpy as np
import cv2
import os

num_pokemons = 809
endpoint = 'https://www.pokemon.com/us/pokedex/'

class ImageProcessor(object):

    def __call__(self, image):
        trans_mask = image[:,:,3] == 0
        image[trans_mask] = [255, 255, 255, 255]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = cv2.resize(image, (image_size, image_size))
        return np.array(image)

class PokemonImageDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.data_root = os.path.join(os.path.dirname(__file__), 'data/images')
        self.transform = transform
        if not os.path.exists(self.data_root):
            if not os.path.exists(os.path.dirname(self.data_root)):
                os.mkdir(os.path.dirname(self.data_root))
            os.mkdir(self.data_root)
            self.extract()
        if len(os.listdir(self.data_root)) < num_pokemons:
            index = len(os.listdir(self.data_root))
            self.extract(start_index=index)

    def extract(self, start_index=1):
        print("extracting...")
        for index in range(start_index, num_pokemons + 1):
            url = endpoint + str(index)
            response = urlopen(url).read()
            soup = BeautifulSoup(response, 'html.parser')
            image_link = soup.select('.profile-images img')[0]['src']
            image = urlopen(image_link).read()
            image_path = os.path.join(self.data_root, "{0:0=3d}.jpg".format(index))
            with open(image_path, 'wb') as output:
                output.write(image)
        print("done extracting")
    def __len__(self):
        return len(os.listdir(self.data_root))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.data_root, os.listdir(self.data_root)[idx])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.transform:
            image = self.transform(image)
        
        return image
    
  