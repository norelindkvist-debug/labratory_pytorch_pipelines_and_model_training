from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision


def get_datasets(transform=None):
    if transform is None:
        transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, transform=transform, download=True
    )
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
