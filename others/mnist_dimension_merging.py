from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

mnist_data = datasets.MNIST('mnist_data/', download=True, train=True)
image = mnist_data[1][0]

ax = plt.gca()
ax.set_xticks([i - 0.5 for i in range(0, 28, 4)])
ax.set_yticks([i - 0.5 for i in range(0, 28, 4)])
ax.set_xticklabels([0, 4, 8, 12, 16, 20, 24])
ax.set_yticklabels([0, 4, 8, 12, 16, 20, 24])
plt.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)

plt.imshow(image, cmap='gray_r')
plt.savefig('./mnist_dimension_merging.pdf')
plt.show()


