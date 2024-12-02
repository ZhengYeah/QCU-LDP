import torch
import torch.nn as nn

# load the model
model = torch.load('ffnn.pth')

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param}")
