import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

x_u = np.linspace(0, 1, 100)
x_v = np.linspace(0, 1, 100)
X_u, X_v = np.meshgrid(x_u, x_v)

h_1 = -0.5 * (X_u ** 2 + X_v ** 2) + np.log(0.5)
h_2 = -0.5 * (0.5 * (X_u - 1) ** 2 + 0.5 * (X_v - 1) ** 2) - 0.5 * np.log(4) + np.log(0.5)
boundary = h_1 - h_2

# Plot the decision regions
plt.figure(figsize=(5, 5))
plt.contour(X_u, X_v, boundary, levels=[0], colors='k')


# add dotted points to distinguish the regions
x_u = np.linspace(0, 1, 60, endpoint=False)
x_v = np.linspace(0, 1, 60, endpoint=False)
X_u, X_v = np.meshgrid(x_u, x_v)

h_1_fine = -0.5 * (X_u ** 2 + X_v ** 2) + np.log(0.5)
h_2_fine = -0.5 * (0.5 * (X_u - 1) ** 2 + 0.5 * (X_v - 1) ** 2) - 0.5 * np.log(4) + np.log(0.5)
boundary_fine = h_1_fine - h_2_fine

# Use a colormap to map decision function values to colors
cmap = mcolors.ListedColormap(['red', 'blue'])
norm = mcolors.BoundaryNorm([-np.inf, 0, np.inf], cmap.N)
plt.scatter(X_u, X_v, c=boundary_fine.flatten(), cmap=cmap, norm=norm, marker='.', s=0.5)

# Remove ending zeros in the ticks
ax = plt.gca()
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

plt.savefig('./example_qda.pdf')
plt.show()
