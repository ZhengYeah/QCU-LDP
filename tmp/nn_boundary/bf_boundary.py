import torch

model = torch.load('ffnn.pth')
grid_2d = torch.rand(1000, 2)
y_pred = model(grid_2d)
y_pred = y_pred.argmax(dim=1)
import matplotlib.pyplot as plt
plt.scatter(grid_2d[:, 0], grid_2d[:, 1], c=y_pred, cmap='coolwarm')
plt.show()
