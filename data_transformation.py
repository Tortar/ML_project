
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2

image_transform_grayscale = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

image_transform_RGB = transforms.Compose([
    v2.Resize(size=(128, 128)),
    transforms.ToTensor(),
])

image_transform_RGB_2 = transforms.Compose([
    v2.Resize(size=(128, 128)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandAugment(),
    transforms.ToTensor(),
])
