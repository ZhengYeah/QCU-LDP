import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add the parent directory to the system path to allow imports from src
BASE_DIR = Path(__file__).resolve().parent.parent # Define the base directory for the project
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20

from src.robust_radius_torch import RobustRadiusTorch
from src.samples_from_mechanism import samples_of_mechanism
from src.cdf_ldp_mechanisms_at_x import CDFAtX


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

model = torch.load(BASE_DIR / 'experiments/mnist/cnn_mnist_7_7.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()
private_image = np.load(BASE_DIR / 'experiments/mnist/mnist_7_7_0.npy')
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
    return prob_accumulated * 0.99 # term (1 - \tau) in the paper, we set tau to 0.01

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
    if mechanism == "laplace" or mechanism == "gaussian":
        accuracy = correct_num / (sample_num + fail_num)
    else:
        accuracy = correct_num / sample_num
    return accuracy

# draw the figure 8 and write the theoretical and empirical accuracy
robust_rectangle = robust_rect()
epsilon_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mechanisms = ["pm", "sw", "exp", "krr"]
theoretical_accuracies = {mechanism: [] for mechanism in mechanisms}
empirical_accuracies = {mechanism: [] for mechanism in mechanisms}
for epsilon in epsilon_values:
    for mechanism in mechanisms:
        prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
        accuracy = empirical_accuracy(epsilon, sample_num=3000, mechanism=mechanism)
        theoretical_accuracies[mechanism].append(0.1 + 0.9 * prob_accumulated)
        empirical_accuracies[mechanism].append(0.1 + 0.9 * accuracy)
# plot the figure
plt.figure()
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(epsilon_values)
plt.plot(epsilon_values, theoretical_accuracies["pm"], label="PM Theoretical", color='red', marker='o')
plt.plot(epsilon_values, empirical_accuracies["pm"], label="PM Empirical", color='red', marker='o', linestyle='--')
plt.fill_between(epsilon_values, theoretical_accuracies["pm"], empirical_accuracies["pm"], color='red', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["sw"], label="SW Theoretical", marker='s', color='black')
plt.plot(epsilon_values, empirical_accuracies["sw"], label="SW Empirical", marker='s', linestyle='--', color='black')
plt.fill_between(epsilon_values, theoretical_accuracies["sw"], empirical_accuracies["sw"], color='black', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["exp"], label="EXP Theoretical", marker='^', color='green')
plt.plot(epsilon_values, empirical_accuracies["exp"], label="EXP Empirical", marker='^', linestyle='--', color='green')
plt.fill_between(epsilon_values, theoretical_accuracies["exp"], empirical_accuracies["exp"], color='green', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["krr"], label="KRR Theoretical", marker='d', color='blue')
plt.plot(epsilon_values, empirical_accuracies["krr"], label="KRR Empirical", marker='d', linestyle='--', color='blue')
plt.fill_between(epsilon_values, theoretical_accuracies["krr"], empirical_accuracies["krr"], color='blue', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon),\hat{\rho}(\varepsilon)$')
plt.legend(fontsize=18)
plt.title('Figure 8(a)')


#########
# Figure 8(b)
#########

private_image = np.load( BASE_DIR / 'experiments/mnist/mnist_7_7_1.npy')
correct_class = model(torch.tensor(private_image).unsqueeze(0).unsqueeze(0).float()).argmax(dim=1)

# draw the figure 8 and write the theoretical and empirical accuracy
robust_rectangle = robust_rect()
epsilon_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mechanisms = ["pm", "sw", "exp", "krr"]
theoretical_accuracies = {mechanism: [] for mechanism in mechanisms}
empirical_accuracies = {mechanism: [] for mechanism in mechanisms}
for epsilon in epsilon_values:
    for mechanism in mechanisms:
        prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
        accuracy = empirical_accuracy(epsilon, sample_num=3000, mechanism=mechanism)
        theoretical_accuracies[mechanism].append(0.1 + 0.9 * prob_accumulated)
        empirical_accuracies[mechanism].append(0.1 + 0.9 * accuracy)
# plot the figure
plt.figure()
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(epsilon_values)
plt.plot(epsilon_values, theoretical_accuracies["pm"], label="PM Theoretical", color='red', marker='o')
plt.plot(epsilon_values, empirical_accuracies["pm"], label="PM Empirical", color='red', marker='o', linestyle='--')
plt.fill_between(epsilon_values, theoretical_accuracies["pm"], empirical_accuracies["pm"], color='red', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["sw"], label="SW Theoretical", marker='s', color='black')
plt.plot(epsilon_values, empirical_accuracies["sw"], label="SW Empirical", marker='s', linestyle='--', color='black')
plt.fill_between(epsilon_values, theoretical_accuracies["sw"], empirical_accuracies["sw"], color='black', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["exp"], label="EXP Theoretical", marker='^', color='green')
plt.plot(epsilon_values, empirical_accuracies["exp"], label="EXP Empirical", marker='^', linestyle='--', color='green')
plt.fill_between(epsilon_values, theoretical_accuracies["exp"], empirical_accuracies["exp"], color='green', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["krr"], label="KRR Theoretical", marker='d', color='blue')
plt.plot(epsilon_values, empirical_accuracies["krr"], label="KRR Empirical", marker='d', linestyle='--', color='blue')
plt.fill_between(epsilon_values, theoretical_accuracies["krr"], empirical_accuracies["krr"], color='blue', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon),\hat{\rho}(\varepsilon)$')
plt.legend(fontsize=18)
plt.title('Figure 8(b)')
plt.show()
