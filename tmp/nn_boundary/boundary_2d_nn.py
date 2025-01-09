import torch
import torch.nn as nn

# load the model
model = torch.load('ffnn.pth')

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param}")

# ========
# Plot the decision boundary
# ========
x = torch.linspace(0, 1, 100)
y = torch.linspace(0, 1, 100)
xx, yy = torch.meshgrid(x, y)
grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)

# pass the grid through the model
with torch.no_grad():
    pred = model(grid)

# plot the decision boundary
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, pred.numpy().reshape(xx.shape), cmap='coolwarm', alpha=0.6)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary')
plt.show()
