from torchvision import datasets
from pathlib import Path

def prepare_mnist():
    data_root = Path("../../../_Data/")

    
    datasets.MNIST(
        root=data_root,
        train=True,
        download=True
    )
    datasets.MNIST(
        root=data_root,
        train=False,
        download=True
    )

if __name__ == "__main__":
    prepare_mnist()