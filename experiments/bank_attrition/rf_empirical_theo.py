from src.samples_from_mechanism import samples_of_mechanism
from src.robust_radius_sklearn import RobustRadiusSKLearn
from src.cdf_ldp_mechanisms_at_x import CDFAtX
import numpy as np
import pandas as pd
import joblib

# private features: age (index 1), salary (index 6)
# choose the last user from the dataset as x
user_row = [619,0.22,2,0.0,1,1,0.5067444,2,464]

private_values = [user_row[1], user_row[6]] # age, salary
x_df = pd.DataFrame(data=[user_row],
                    columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember',
                             'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
model = joblib.load('classifiers/bank_rf.pkl')


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


if __name__ == '__main__':
    robust_rectangle = robust_rect_rf()

    # write the theoretical and empirical accuracy to csv file
    with open('rf_accuracy.csv', 'w') as f:
        f.write('epsilon,pm_theo,pm_empirical,exp_theo,exp_empirical,laplace_theo,laplace_empirical,gaussian_theo,gaussian_empirical\n')
        for epsilon in range(1, 9):
            f.write(f'{epsilon}')
            for mechanism in ["pm", "exp", "laplace", "gaussian"]:
                prob_accumulated = theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
                accuracy = empirical_accuracy(epsilon, sample_num=3000, mechanism=mechanism)
                f.write(f',{prob_accumulated:.6f},{accuracy:.3f}')
            f.write('\n')

