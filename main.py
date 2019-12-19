import pokegan
import torch


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gan = pokegan.GAN(device)
    gan.train(epochs=3000)
    gan.save()