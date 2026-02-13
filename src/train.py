import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model import SimpleClassifier
from utils import Device
from datasets import get_datasets, get_dataloaders

device = Device()

base_transform = transforms.ToTensor()
train_dataset, test_dataset = get_datasets(transform=base_transform)

imgs = torch.stack([img for img, _ in train_dataset], dim=0)
mean = imgs.mean(dim=(0, 2, 3))
std  = imgs.std(dim=(0, 2, 3))

cifar_mean = tuple(mean.tolist())
cifar_std  = tuple(std.tolist())
print("Mean:", cifar_mean)
print("Std :", cifar_std)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

train_dataset, test_dataset = get_datasets(transform=transform)

train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=64)

model = SimpleClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def evaluate(model, loader):
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

epochs = 10
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluate(model, train_loader)
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}/{epochs} | train acc {train_acc:.3f} loss {train_loss:.3f} | test acc {test_acc:.3f} loss {test_loss:.3f}")
