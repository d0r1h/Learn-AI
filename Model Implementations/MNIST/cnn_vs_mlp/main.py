import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from models.mlp import MLP
from models.cnn import CNN
from data.mnist import get_mnist_loaders
from trainers.trainer import Trainer


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["system"]["num_workers"]
    )

    if cfg["model"]["name"] == "mlp":
        model = MLP().to(device)
    else:
        model = CNN().to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=cfg["training"]["lr"]
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, device)

    for epoch in range(cfg["training"]["epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        test_acc = trainer.evaluate(test_loader)

        print(
            f"Epoch [{epoch+1}/{cfg['training']['epochs']}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )


if __name__ == "__main__":
    main()