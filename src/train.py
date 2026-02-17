import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.model import SimpleClassifier
from src.utils import Device
from src.datasets import get_datasets, get_dataloaders


def train_model(epochs=10, batch_size=64, lr=1e-3):
    device = Device()

    base_transform = transforms.ToTensor()
    train_dataset, test_dataset = get_datasets(transform=base_transform)

    imgs = torch.stack([img for img, _ in train_dataset], dim=0)
    mean = imgs.mean(dim=(0, 2, 3))
    std = imgs.std(dim=(0, 2, 3))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(tuple(mean.tolist()), tuple(std.tolist())),
        ]
    )

    train_dataset, test_dataset = get_datasets(transform=transform)

    train_loader, test_loader = get_dataloaders(
        train_dataset, test_dataset, batch_size=batch_size
    )

    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

        #keep last metrics so we can return them to main.py
    last_train_loss = last_train_acc = None
    last_test_loss = last_test_acc = None


    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        last_train_loss, last_train_acc = evaluate(model, train_loader, criterion, device)
        last_test_loss, last_test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train acc {last_train_acc:.3f} loss {last_train_loss:.3f} | "
            f"test acc {last_test_acc:.3f} loss {last_test_loss:.3f}"
        )

        #I want to save result in a result.md file so I can look at the results later if I want. 
        results = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_loss": last_train_loss,
        "train_acc": last_train_acc,
        "test_loss": last_test_loss,
        "test_acc": last_test_acc,
    }
    return model, results

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total
