
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Dataset location
DATA_DIR = Path(__file__).resolve().parent / "data"

# Image transformation
transform = transforms.ToTensor()

# Download training dataset (60,000 images)
train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

# Download testing dataset (10,000 images)
test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


if __name__ == "__main__":

    images, labels = next(iter(train_loader))

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)

    print("First image shape:", images[0].shape)
    print("First image label:", labels[0].item())

    print("Minimum pixel value:", images.min().item())
    print("Maximum pixel value:", images.max().item())
