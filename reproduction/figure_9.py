import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add the parent directory to the system path to allow imports from src
BASE_DIR = Path(__file__).resolve().parent.parent # Define the base directory for the project
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20

from src.samples_from_mechanism import samples_of_mechanism
from src.robust_radius_sklearn import RobustRadiusSKLearn
from src.cdf_ldp_mechanisms_at_x import CDFAtX

# private features: age (index 0), bmi (index 5), be cautious about the index
# the first user
user_row = [0.67,0.0,1.0,1.0,0.7623,0.732]
private_ind_1, private_ind_2 = 0, 5

private_values = [user_row[private_ind_1], user_row[private_ind_2]] # age, bmi
data_columns = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi']
x_df = pd.DataFrame(data=[user_row], columns=data_columns)
model = joblib.load(BASE_DIR / 'experiments/stroke_pred/classifiers/stroke_lr.pkl')


def robust_rect_rf():
    robust_rec = RobustRadiusSKLearn(model, x_df, ['age','bmi'], 0.04, 0.01)
    radius = robust_rec.binary_search()
    robust_rectangle = robust_rec.adjust_step_rate([(0.4, 0.5), (0.3, 0.5), (0.2, 0.5)])
    robust_rectangle = robust_rec.adjust_step_rate([(0.1, 0.5), (0.05, 0.5)])
    return robust_rectangle

def theoretical_accuracy(epsilon, robust_rectangle, mechanism="pm"):
    # compute the theoretical accuracy
    prob_accumulated = 1
    for i, private_value in enumerate(private_values):
        cdf_at_x = CDFAtX(epsilon, private_value, bin_num=100)
        rectangle = [robust_rectangle[0][i], robust_rectangle[1][i]]
        cdf_rect = cdf_at_x.cdf_of_tilde_x(rectangle, mechanism)
        prob_accumulated = prob_accumulated * cdf_rect * 0.99 # term (1 - \tau) in the paper, we set tau to 0.01
    return prob_accumulated * 0.99 # term (1 - \tau) in the paper, we set tau to 0.01


def empirical_accuracy(epsilon, sample_num=3000, mechanism="pm"):
    if mechanism == "laplace" or mechanism == "gaussian":
        samples, fail_num_laplace = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)
    else:
        samples = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)

    perturbed_age = samples[:,0]
    perturbed_bmi = samples[:, 1]
    # fill the perturbed rows with the original row
    perturbed_rows = user_row * np.ones((sample_num, len(user_row)))
    # NOTE: index 0 is age, index 5 is bmi
    perturbed_rows[:,private_ind_1] = perturbed_age
    perturbed_rows[:,private_ind_2] = perturbed_bmi
    # form dataframes
    # CreditScore,Age,Tenure,Balance,NumOfProducts,IsActiveMember,EstimatedSalary,Exited,Satisfaction Score,Point Earned
    perturbed_df = pd.DataFrame(data=perturbed_rows, columns=data_columns)
    # feed to the trained model
    ground_truth = model.predict(x_df)
    pred = model.predict(perturbed_df)
    # calculate the empirical accuracy
    if mechanism == "laplace" or mechanism == "gaussian":
        accuracy = np.sum(ground_truth == pred) / (sample_num + fail_num_laplace)
    else:
        accuracy = np.sum(ground_truth == pred) / sample_num
    return accuracy

robust_rectangle = robust_rect_rf()
epsilon_values = range(1, 9)
mechanisms = ["pm", "sw", "krr", "exp", "laplace", "gaussian"]
theoretical_accuracies = {mechanism: [] for mechanism in mechanisms}
empirical_accuracies = {mechanism: [] for mechanism in mechanisms}
for epsilon in epsilon_values:
    for mechanism in mechanisms:
        prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
        accuracy = empirical_accuracy(epsilon, sample_num=6000, mechanism=mechanism)
        theoretical_accuracies[mechanism].append(prob_accumulated)
        empirical_accuracies[mechanism].append(accuracy)
# plot the figure
plt.figure()
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(epsilon_values)
# shaded region between the theoretical and empirical accuracy
plt.plot(epsilon_values, theoretical_accuracies["sw"], label="SW", marker='o', color='red', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["sw"], marker='o', linestyle='--', color='red')
plt.fill_between(epsilon_values, theoretical_accuracies["sw"], empirical_accuracies["sw"], color='red', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["krr"], label="k-RR", marker='x', color='green', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["krr"], marker='x', linestyle='--', color='green')
plt.fill_between(epsilon_values, theoretical_accuracies["krr"], empirical_accuracies["krr"], color='green', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon), \hat{\rho}(\varepsilon)$')
plt.title('Figure 9a')
plt.legend(fontsize=18)

########
# Figure 9b
########

model = joblib.load(BASE_DIR / 'experiments/stroke_pred/classifiers/stroke_rf.pkl')

robust_rectangle = robust_rect_rf()
epsilon_values = range(1, 9)
mechanisms = ["pm", "sw", "krr", "exp", "laplace", "gaussian"]
theoretical_accuracies = {mechanism: [] for mechanism in mechanisms}
empirical_accuracies = {mechanism: [] for mechanism in mechanisms}
for epsilon in epsilon_values:
    for mechanism in mechanisms:
        prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
        accuracy = empirical_accuracy(epsilon, sample_num=6000, mechanism=mechanism)
        theoretical_accuracies[mechanism].append(prob_accumulated)
        empirical_accuracies[mechanism].append(accuracy)
# plot the figure
plt.figure()
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(epsilon_values)
# shaded region between the theoretical and empirical accuracy
plt.plot(epsilon_values, theoretical_accuracies["sw"], label="SW", marker='o', color='red', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["sw"], marker='o', linestyle='--', color='red')
plt.fill_between(epsilon_values, theoretical_accuracies["sw"], empirical_accuracies["sw"], color='red', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["krr"], label="k-RR", marker='x', color='green', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["krr"], marker='x', linestyle='--', color='green')
plt.fill_between(epsilon_values, theoretical_accuracies["krr"], empirical_accuracies["krr"], color='green', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon), \hat{\rho}(\varepsilon)$')
plt.title('Figure 9b')
plt.legend(fontsize=18)
plt.show()

