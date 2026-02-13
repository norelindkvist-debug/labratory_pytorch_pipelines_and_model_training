from src.datasets import get_dataloaders


def main():
    train_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
    print("Image shape:", images.shape)


if __name__ == "__main__":
    main()
