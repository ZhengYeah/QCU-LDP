import torch
import torch.nn as nn

# load the model
model = torch.load('ffnn.pth')

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param}")

# ========
# Plot the decision boundary
# ========
x = torch.linspace(0, 1, 200)
y = torch.linspace(0, 1, 200)
xx, yy = torch.meshgrid(x, y, indexing='ij')
grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)

# pass the grid through the model
with torch.no_grad():
    pred = model(grid).argmax(dim=1)

# reshape the prediction and change to python scalar
pred = pred.view(xx.size()).numpy()

# plot the decision boundary
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.contourf(xx.numpy(), yy.numpy(), pred, cmap='coolwarm', alpha=0.5)
# emphisize the decision boundary
plt.contour(xx.numpy(), yy.numpy(), pred, levels=[0.5], colors='black', linestyles='--')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary')
plt.show()
