from src.robust_radius_torch import RobustRadiusTorch
from src.merge_dim_of_2d_img import merge_dim_of_2d_img
from src.samples_from_mechanism import samples_of_mechanism
from src.cdf_ldp_mechanisms_at_x import CDFAtX
import numpy as np
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = torch.load('cnn_mnist_14_14.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()
private_image = np.load('mnist_14_14.npy')
correct_class = model(torch.tensor(private_image).unsqueeze(0).unsqueeze(0).float()).argmax(dim=1)

merged_ind, unmerged_ind, grid_step = merge_dim_of_2d_img(private_image, twice_grid_step=2)
# transform to tensor
rows_merged = torch.tensor([merged_ind[i][0] for i in range(len(merged_ind))])
cols_merged = torch.tensor([merged_ind[i][1] for i in range(len(merged_ind))])
tensor_merged_ind = torch.stack((rows_merged, cols_merged), dim=0)
rows_unmerged = torch.tensor([unmerged_ind[i][0] for i in range(len(unmerged_ind))])
cols_unmerged = torch.tensor([unmerged_ind[i][1] for i in range(len(unmerged_ind))])
tensor_unmerged_ind = torch.stack((rows_unmerged, cols_unmerged), dim=0)


def robust_rect():
    x = torch.tensor(private_image, dtype=torch.float32)
    robust_rec = RobustRadiusTorch(model, x, merged_ind, unmerged_ind, grid_step, 0.05, 0.1)
    radius = robust_rec.binary_search()
    print(f"Robust radius: {radius}")
    founded_rec = robust_rec.adjust_step_rate([(0.04, 0.5), (0.04, 0.2), (0.03, 0.5), (0.03, 0.2)])
    refined_rec = robust_rec.adjust_step_rate([(0.02, 0.1), (0.01, 0.1)])
    print(f"Founded robust rectangle: {founded_rec}")
    print(f"Refined robust rectangle: {refined_rec}")
    return refined_rec

def theoretical_accuracy(epsilon, robust_rectangle, mechanism="pm"):
    # compute the theoretical accuracy
    prob_accumulated = 1
    # shaper of the robust rectangle: (2, 14, 14)
    assert robust_rectangle[0].shape == robust_rectangle[1].shape
    for i in range (robust_rectangle[0].shape[0]):
        for j in range (robust_rectangle[0].shape[1]):
            rectangle = (robust_rectangle[0][i, j], robust_rectangle[1][i, j])
            cdf_at_x = CDFAtX(epsilon, private_image[i, j])
            cdf_rect = cdf_at_x.cdf_of_tilde_x(rectangle, mechanism)
            # NOTE: calibration is necessary due to the large error accumulation
            cdf_rect = 1 if abs(1 - cdf_rect) < 2e-2 else cdf_rect
            prob_accumulated *= cdf_rect
    return prob_accumulated

def empirical_accuracy(epsilon, sample_num=3000, mechanism="pm"):
    flatten_private_image = private_image.flatten()
    if mechanism == "laplace":
        samples, fail_num_laplace = samples_of_mechanism(flatten_private_image, sample_num, mechanism, epsilon)
    else:
        samples = samples_of_mechanism(flatten_private_image, sample_num, mechanism, epsilon)
    # process the merged dims
    samples = torch.tensor(samples, dtype=torch.float32)
    samples = samples.view(sample_num, private_image.shape[0], private_image.shape[1])
    for i in range(1, grid_step):
        samples[:, rows_merged + i, cols_merged] = samples[:, rows_merged, cols_merged]
        samples[:, rows_merged, cols_merged + i] = samples[:, rows_merged, cols_merged]
        samples[:, rows_merged + i, cols_merged + i] = samples[:, rows_merged, cols_merged]
    with torch.no_grad():
        pred = model(samples)
    pred = pred.argmax(dim=1)
    # ground truth
    # calculate the empirical accuracy
    correct_num = (pred == correct_class).sum().item()
    if mechanism == "laplace":
        accuracy = correct_num / (sample_num + fail_num_laplace)
    else:
        accuracy = correct_num / sample_num
    return accuracy


if __name__ == '__main__':
    epsilon = 4
    robust_rectangle = robust_rect()
    theoretical_acc = theoretical_accuracy(epsilon, robust_rectangle)
    print(f"Theoretical accuracy: {theoretical_acc}")
    # empirical_acc = empirical_accuracy(epsilon)
    # print(f"Empirical accuracy: {empirical_acc}")
