import torch
from pathlib import Path

def Device():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
    return torch.device(device)

DATA_PATH = Path(__file__).parent / "data"