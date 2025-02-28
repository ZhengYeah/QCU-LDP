from src.robust_radius_torch import RobustRadiusTorch
from src.depracated.merge_dim_of_2d_img import merge_dim_of_2d_img
import numpy as np
import torch

private_image = np.load('../experiments/mnist/mnist_14_14.npy')
merged_dims, unmerged_dims, grid_step = merge_dim_of_2d_img(private_image, twice_grid_step=2)
model = torch.load('../experiments/mnist/ffnn_mnist_14_14.pth', map_location=torch.device('cpu'), weights_only=False)


def test_robust_radius():
    x = torch.tensor(private_image, dtype=torch.float32)
    robust_radius = RobustRadiusTorch(model, x, merged_dims, unmerged_dims, grid_step, 0.05, 0.05)
    radius = robust_radius.binary_search()
    assert radius > 0
    print(f"Robust radius: {radius}")

def test_robust_rect():
    x = torch.tensor(private_image, dtype=torch.float32)
    robust_rec = RobustRadiusTorch(model, x, merged_dims, unmerged_dims, grid_step, 0.05, 0.05)
    radius = robust_rec.binary_search()
    print(f"Robust radius: {radius}")
    founded_rec = robust_rec.adjust_step_rate([(0.3, 0.5), (0.2, 0.5), (0.1, 0.5), (0.05, 0.5)])
    refined_rec = robust_rec.adjust_step_rate([(0.1, 0.5), (0.05, 0.5)])
    print(f"Founded robust rectangle: {founded_rec}")
    print(f"Refined robust rectangle: {refined_rec}")
