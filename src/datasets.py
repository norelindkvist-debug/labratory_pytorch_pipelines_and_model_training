from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

def get_dataloaders(batch_size=64):

    train_dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader