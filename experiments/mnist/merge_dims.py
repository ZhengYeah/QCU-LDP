from src.merge_dim_of_2d_img import merge_dim_of_2d_img
import numpy as np

# load the image
image = np.load('mnist_14_14.npy')
merged_dim = merge_dim_of_2d_img(image, twice_grid_step=2)
print(f"The merged dimension is {merged_dim} with length {len(merged_dim)}")
