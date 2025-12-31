from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
from pathlib import Path

image_path = Path('../../../_Data')

def get_mnist_loaders(batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root = image_path,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root = image_path,
        train=False,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader