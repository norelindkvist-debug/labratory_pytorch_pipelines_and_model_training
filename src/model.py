import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(3072, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten image

        return self.model(x)
