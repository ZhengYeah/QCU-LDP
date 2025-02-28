from src.robust_radius_torch import RobustRadiusTorch
from src.merge_dim_of_2d_img import merge_dim_of_2d_img
from src.samples_from_mechanism import samples_of_mechanism
from src.cdf_ldp_mechanisms_at_x import CDFAtX
import numpy as np
import torch
import torch.nn as nn

private_image = np.load('mnist_14_14.npy')
merged_dims, unmerged_dims, grid_step = merge_dim_of_2d_img(private_image, twice_grid_step=2)
model = torch.load('ffnn_mnist_14_14.pth', map_location=torch.device('cpu'), weights_only=False)


def robust_rect():
    x = torch.tensor(private_image, dtype=torch.float32)
    robust_rec = RobustRadiusTorch(model, x, merged_dims, unmerged_dims, grid_step, 0.05, 0.05)
    radius = robust_rec.binary_search()
    print(f"Robust radius: {radius}")
    founded_rec = robust_rec.adjust_step_rate([(0.4, 0.5), (0.4, 0.4), (0.3, 0.5), (0.4, 0.3), (0.3, 0.4), (0.2, 0.5), (0.5, 0.2)])
    refined_rec = robust_rec.adjust_step_rate([(0.1, 0.8), (0.1, 0.5), (0.05, 0.5)])
    print(f"Founded robust rectangle: {founded_rec}")
    print(f"Refined robust rectangle: {refined_rec}")
    return refined_rec

def theoretical_accuracy(epsilon, robust_rectangle, mechanism="pm"):
    # compute the theoretical accuracy
    prob_accumulated = 1
    for i, merged_dim in enumerate(merged_dims):
        cdf_at_x = CDFAtX(epsilon, private_image[merged_dim], bin_num=256)
        rectangle = [robust_rectangle[0][i], robust_rectangle[1][i]]
        cdf_rect = cdf_at_x.cdf_of_tilde_x(rectangle, mechanism)
        prob_accumulated *= cdf_rect
    return prob_accumulated

def empirical_accuracy(epsilon, sample_num=1000, mechanism="pm"):
    if mechanism == "laplace":
        samples, fail_num_laplace = samples_of_mechanism(private_image, sample_num, mechanism, epsilon)
    else:
        samples = samples_of_mechanism(private_image, sample_num, mechanism, epsilon)

    perturbed_df = np.zeros((sample_num, len(private_image)))
    for i, merged_dim in enumerate(merged_dims):
        perturbed_df[:, merged_dim] = samples[:, i]
    # feed to the trained model
    ground_truth = model(torch.tensor(private_image).unsqueeze(0))
    pred = model(torch.tensor(perturbed_df))
    # calculate the empirical accuracy
    if mechanism == "laplace":
        accuracy = np.sum(ground_truth == pred) / (sample_num + fail_num_laplace)
    else:
        accuracy = np.sum(ground_truth == pred) / sample_num
    return accuracy


if __name__ == '__main__':
    robust_rectangle = robust_rect()
    epsilon = 1
    theoretical_acc = theoretical_accuracy(epsilon, robust_rectangle)
    print(f"Theoretical accuracy: {theoretical_acc}")
