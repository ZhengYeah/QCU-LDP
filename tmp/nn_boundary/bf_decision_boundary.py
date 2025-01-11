import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20


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
plt.figure(figsize=(8, 8))
plt.contourf(xx.numpy(), yy.numpy(), pred, cmap='coolwarm', alpha=0.5)
# emphisize the decision boundary
plt.contour(xx.numpy(), yy.numpy(), pred, levels=[0.5], colors='black', linestyles='--')
plt.title('Decision Boundary')
# Remove ending zeros in the ticks
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.savefig('./decision_boundary.pdf')
plt.show()
