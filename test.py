from pokegan.data import PokemonImageDataset, ImageProcessor
from pokegan import Generator
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def view_train_images():
    index = 0
    while index < end_index:  # the following below is part of the transformation for actual batches
        index += 1
        image_path = './pokegan/data/images/' + "{0:0=3d}.jpg".format(index)
        test_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        trans_mask = test_image[:,:,3] == 0
        test_image[trans_mask] = [255, 255, 255, 255]
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGRA2BGR)
        test_image = cv2.resize(test_image, (100, 100))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        plt.imshow(test_image)
        plt.show()

def view_batch_samples():
    data_set = PokemonImageDataset(transform=transforms.Compose([
        ImageProcessor(),
        #transforms.ToTensor()
    ]))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=16, shuffle=True)
    batch = next(iter(data_loader))
    plt.imshow(batch[0])
    plt.show()



if __name__ == "__main__":
    model = Generator()
    model.load_state_dict(torch.load('./PokeGAN/generator.pt'))
    model.eval()
    tensor = torch.randn(5, 100, 1, 1)
    output = model(tensor)
    image = output[3] / torch.max(output)
    print(image.min())
    image = np.transpose(image.detach(), (1,2,0))
    plt.imshow(image)
    plt.show()
    
