from src.samples_from_mechanism import samples_of_mechanism
from src.robust_radius_sklearn import RobustRadiusSKLearn
from src.cdf_ldp_mechanisms_at_x import CDFAtX
import numpy as np
import pandas as pd
import joblib

# private features: age (index 1), salary (index 6)
# choose a user from the dataset as x
user_row = [0.619,0.22,0.2,0.0,0.25,1.0,0.5067444,0.4,0.464]

private_values = [user_row[1], user_row[6]] # age, salary
x_df = pd.DataFrame(data=[user_row],
                    columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember',
                             'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
model = joblib.load('../experiments/bank_attrition/classifiers/bank_lr.pkl')


def robust_rect_rf():
    robust_rec = RobustRadiusSKLearn(model, x_df, ['Age','EstimatedSalary'], 0.05, 0.05)
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
        prob_accumulated *= cdf_rect
    return prob_accumulated


def empirical_accuracy(epsilon, sample_num=4000, mechanism="pm"):
    if mechanism == "laplace" or mechanism == "gaussian":
        samples, fail_num = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)
    else:
        samples = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)

    perturbed_age = samples[:,0]
    perturbed_salary = samples[:,1]
    # fill the perturbed rows with the original row
    perturbed_rows = user_row * np.ones((sample_num, len(user_row)))
    perturbed_rows[:,1] = perturbed_age
    perturbed_rows[:,6] = perturbed_salary
    # form dataframes
    # CreditScore,Age,Tenure,Balance,NumOfProducts,IsActiveMember,EstimatedSalary,Exited,Satisfaction Score,Point Earned
    perturbed_df = pd.DataFrame(data=perturbed_rows, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
    # feed to the trained model
    ground_truth = model.predict(x_df)
    pred = model.predict(perturbed_df)
    # calculate the empirical accuracy
    if mechanism == "laplace" or mechanism == "gaussian":
        accuracy = np.sum(ground_truth == pred) / (sample_num + fail_num)
    else:
        accuracy = np.sum(ground_truth == pred) / sample_num
    return accuracy


# draw the figure of theoretical and empirical accuracy for different mechanisms and different epsilon values
robust_rectangle = robust_rect_rf()
epsilon_values = range(1, 9)
mechanisms = ["pm", "sw", "krr", "exp", "laplace", "gaussian"]
theoretical_accuracies = {mechanism: [] for mechanism in mechanisms}
empirical_accuracies = {mechanism: [] for mechanism in mechanisms}
for epsilon in epsilon_values:
    for mechanism in mechanisms:
        prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
        accuracy = empirical_accuracy(epsilon, sample_num=6000, mechanism=mechanism)
        if mechanism == "gaussian":
            theoretical_accuracies[mechanism].append(prob_accumulated)
            empirical_accuracies[mechanism].append(accuracy)
        else:
            theoretical_accuracies[mechanism].append(0.1 + 0.9 * prob_accumulated)
            empirical_accuracies[mechanism].append(0.1 + 0.9 * accuracy)
# plot the figure
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(epsilon_values)
plt.plot(epsilon_values, theoretical_accuracies["pm"], label="PM", marker='o', color='red', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["pm"], marker='o', linestyle='--', color='red')
plt.fill_between(epsilon_values, theoretical_accuracies["pm"], empirical_accuracies["pm"], color='red', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["exp"], label="Exp", marker='x', color='green', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["exp"], marker='x', linestyle='--', color='green')
plt.fill_between(epsilon_values, theoretical_accuracies["exp"], empirical_accuracies["exp"], color='green', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["laplace"], label="Laplace", marker='s', color='blue', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["laplace"], marker='s', linestyle='--', color='blue')
plt.fill_between(epsilon_values, theoretical_accuracies["laplace"], empirical_accuracies["laplace"], color='blue', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["gaussian"], label="Gaussian", marker='d', color='purple', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["gaussian"], marker='d', linestyle='--', color='purple')
plt.fill_between(epsilon_values, theoretical_accuracies["gaussian"], empirical_accuracies["gaussian"], color='purple', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=18)
plt.title('Figure 7(a)')
plt.show()


########
# Figure 7(b)
########

model = joblib.load('../experiments/bank_attrition/classifiers/bank_rf.pkl')

def robust_rect_rf():
    robust_rec = RobustRadiusSKLearn(model, x_df, ['Age','EstimatedSalary'], 0.05, 0.05)
    radius = robust_rec.binary_search()
    robust_rectangle = robust_rec.adjust_step_rate([(0.4, 0.5), (0.3, 0.5), (0.2, 0.5)])
    robust_rectangle = robust_rec.adjust_step_rate([(0.1, 0.5), (0.05, 0.5)])
    return robust_rectangle

def empirical_accuracy(epsilon, sample_num=5000, mechanism="pm"):
    if mechanism == "laplace" or mechanism == "gaussian":
        samples, fail_num = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)
    else:
        samples = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)

    perturbed_age = samples[:,0]
    perturbed_salary = samples[:,1]
    # fill the perturbed rows with the original row
    perturbed_rows = user_row * np.ones((sample_num, len(user_row)))
    perturbed_rows[:,1] = perturbed_age
    perturbed_rows[:,6] = perturbed_salary
    # form dataframes
    # CreditScore,Age,Tenure,Balance,NumOfProducts,IsActiveMember,EstimatedSalary,Exited,Satisfaction Score,Point Earned
    perturbed_df = pd.DataFrame(data=perturbed_rows, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
    # feed to the trained model
    ground_truth = model.predict(x_df)
    pred = model.predict(perturbed_df)
    # calculate the empirical accuracy
    if mechanism == "laplace" or mechanism == "gaussian":
        accuracy = np.sum(ground_truth == pred) / (sample_num + fail_num)
    else:
        accuracy = np.sum(ground_truth == pred) / sample_num
    return accuracy

# draw the figure of theoretical and empirical accuracy for different mechanisms and different epsilon values
robust_rectangle = robust_rect_rf()
epsilon_values = range(1, 9)
mechanisms = ["pm", "sw", "krr", "exp", "laplace", "gaussian"]
theoretical_accuracies = {mechanism: [] for mechanism in mechanisms}
empirical_accuracies = {mechanism: [] for mechanism in mechanisms}
for epsilon in epsilon_values:
    for mechanism in mechanisms:
        prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
        accuracy = empirical_accuracy(epsilon, sample_num=6000, mechanism=mechanism)
        if mechanism == "gaussian":
            theoretical_accuracies[mechanism].append(prob_accumulated)
            empirical_accuracies[mechanism].append(accuracy)
        else:
            theoretical_accuracies[mechanism].append(0.1 + 0.9 * prob_accumulated)
            empirical_accuracies[mechanism].append(0.1 + 0.9 * accuracy)
# plot the figure
plt.rcParams['font.size'] = 20
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(epsilon_values)
plt.plot(epsilon_values, theoretical_accuracies["pm"], label="PM", marker='o', color='red', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["pm"], marker='o', linestyle='--', color='red')
plt.fill_between(epsilon_values, theoretical_accuracies["pm"], empirical_accuracies["pm"], color='red', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["exp"], label="Exp", marker='x', color='green', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["exp"], marker='x', linestyle='--', color='green')
plt.fill_between(epsilon_values, theoretical_accuracies["exp"], empirical_accuracies["exp"], color='green', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["laplace"], label="Laplace", marker='s', color='blue', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["laplace"], marker='s', linestyle='--', color='blue')
plt.fill_between(epsilon_values, theoretical_accuracies["laplace"], empirical_accuracies["laplace"], color='blue', alpha=0.08)
plt.plot(epsilon_values, theoretical_accuracies["gaussian"], label="Gaussian", marker='d', color='purple', linewidth=2.5)
plt.plot(epsilon_values, empirical_accuracies["gaussian"], marker='d', linestyle='--', color='purple')
plt.fill_between(epsilon_values, theoretical_accuracies["gaussian"], empirical_accuracies["gaussian"], color='purple', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=18)
plt.title('Figure 7(b)')
plt.show()
