from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

mnist_data = datasets.MNIST('mnist_data/', download=False, train=True)
image = mnist_data[1][0] # get the second image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=4, stride=4))
])

image = transform(image)
image = image.squeeze().numpy()
# save the image as numpy array
np.save('mnist_7_7.npy', image)

# ax = plt.gca()
# ax.set_xticks([i - 0.5 for i in range(0, 14, 2)])
# ax.set_yticks([i - 0.5 for i in range(0, 14, 2)])
# ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12])
# ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12])
# plt.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)

ax = plt.gca()
ax.set_xticks([i - 0.5 for i in range(0, 7)])
ax.set_yticks([i - 0.5 for i in range(0, 7)])
plt.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)

plt.imshow(image, cmap='gray_r')
plt.show()


