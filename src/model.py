import torch.nn as nn

input_size = 1024
output_size = 10 
hidden_layers = 128

model = nn.Sequential(
    nn.Linear(input_size, hidden_layers),
    nn.ReLU(),
    nn.Linear(hidden_layers, output_size)
)