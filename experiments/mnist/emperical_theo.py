from src.robust_radius_torch import RobustRadiusTorch
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
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = torch.load('cnn_mnist_7_7.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()
private_image = np.load('mnist_7_7_0.npy')
correct_class = model(torch.tensor(private_image).unsqueeze(0).unsqueeze(0).float()).argmax(dim=1)

def robust_rect():
    x = torch.tensor(private_image, dtype=torch.float32)
    robust_rec = RobustRadiusTorch(model, x, 0.05, 0.1)
    radius = robust_rec.binary_search()
    print(f"Robust radius: {radius}")
    # founded_rec = robust_rec.adjust_step_rate([(0.04, 0.5), (0.04, 0.2), (0.03, 0.5), (0.03, 0.2)])
    refined_rec = robust_rec.adjust_step_rate([(0.02, 0), (0.01, 0.1)])
    # print(f"Founded robust rectangle: {founded_rec}")
    print(f"Refined robust rectangle:\n{refined_rec[0]}\n{refined_rec[1]}")
    return refined_rec

def theoretical_accuracy(epsilon, robust_rectangle, mechanism="pm"):
    # compute the theoretical accuracy
    prob_accumulated = 1
    # shaper of the robust rectangle: (2, 7, 7)
    assert robust_rectangle[0].shape == robust_rectangle[1].shape
    for i in range (robust_rectangle[0].shape[0]):
        for j in range (robust_rectangle[0].shape[1]):
            rectangle = (robust_rectangle[0][i, j], robust_rectangle[1][i, j])
            cdf_at_x = CDFAtX(epsilon, private_image[i, j], bin_num=100)
            cdf_rect = cdf_at_x.cdf_of_tilde_x(rectangle, mechanism)
            cdf_rect = 1 if abs(1 - cdf_rect) < 1e-2 else cdf_rect
            prob_accumulated *= cdf_rect
    return prob_accumulated

def empirical_accuracy(epsilon, sample_num=3000, mechanism="pm"):
    flatten_private_image = private_image.flatten()
    if mechanism == "laplace" or mechanism == "gaussian":
        samples, fail_num = samples_of_mechanism(flatten_private_image, sample_num, mechanism, epsilon)
    else:
        samples = samples_of_mechanism(flatten_private_image, sample_num, mechanism, epsilon)
    # process the merged dims
    samples = torch.tensor(samples, dtype=torch.float32)
    samples = samples.view(sample_num, private_image.shape[0], private_image.shape[1])
    # unsqueeze the samples for CNN
    samples = samples.unsqueeze(1).float()
    with torch.no_grad():
        pred = model(samples)
    pred = pred.argmax(dim=1)
    # ground truth
    # calculate the empirical accuracy
    correct_num = (pred == correct_class).sum().item()
    if mechanism == "laplace":
        accuracy = correct_num / (sample_num + fail_num)
    else:
        accuracy = correct_num / sample_num
    return accuracy


if __name__ == '__main__':
    robust_rectangle = robust_rect()
    # write the theoretical and empirical accuracy to csv file
    with open('cnn_accuracy_0.csv', 'w') as f:
        f.write('epsilon,pm_theo,pm_empirical,sw_theo,sw_empirical,exp_theo,exp_empirical,krr_theo,krr_empirical\n')
        for epsilon in range(1, 9):
            f.write(f'{epsilon}')
            for mechanism in ["pm", "sw", "exp", "krr"]:
                prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
                # repeat 3 times to get the std
                empirical_accuracies = empirical_accuracy(epsilon, sample_num=3000, mechanism=mechanism)
                f.write(f',{prob_accumulated:.6f},{empirical_accuracies:.3f}')
            f.write('\n')
