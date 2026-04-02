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

# private features: age (index 1), salary (index 6)
private_ind_1, private_ind_2 = 1, 6
data_columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember',
                             'EstimatedSalary', 'Satisfaction Score', 'Point Earned']
model = joblib.load(BASE_DIR / 'experiments/bank_attrition/classifiers/bank_lr.pkl')

def robust_rect_rf(x_dataframe):
    robust_rec = RobustRadiusSKLearn(model, x_dataframe, ['Age','EstimatedSalary'], 0.05, 0.03)
    radius = robust_rec.binary_search()
    robust_rectangle = robust_rec.adjust_step_rate([(0.4, 0.5), (0.3, 0.5), (0.2, 0.5)])
    robust_rectangle = robust_rec.adjust_step_rate([(0.1, 0.5), (0.05, 0.5)])
    return robust_rectangle

def theoretical_accuracy(private_values, epsilon, robust_rectangle, mechanism="pm"):
    # compute the theoretical accuracy
    prob_accumulated = 1
    for i, private_value in enumerate(private_values):
        cdf_at_x = CDFAtX(epsilon, private_value, bin_num=100)
        rectangle = [robust_rectangle[0][i], robust_rectangle[1][i]]
        cdf_rect = cdf_at_x.cdf_of_tilde_x(rectangle, mechanism)
        prob_accumulated *= cdf_rect
    return prob_accumulated * 0.99 # term (1 - \tau) in the paper, we set tau to 0.01


def empirical_accuracy(user_row, epsilon, sample_num=3000, mechanism="pm"):
    private_values = [user_row[private_ind_1], user_row[private_ind_2]]
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


if __name__ == '__main__':
    # load the processed stroke data, without index
    df = pd.read_csv(BASE_DIR / 'experiments/bank_attrition/classifiers/processed_bank_data_normalized.csv')
    df = df.iloc[:100].drop(columns=['Exited'])
    with open('lr_avg_wor.csv', 'w') as f:
        f.write('x_pv_1,x_pv_2,epsilon,pm_theo,pm_empirical,sw_theo,sw_empirical,krr_theo,krr_empirical,exp_theo,exp_empirical,laplace_theo,laplace_empirical,gaussian_theo,gaussian_empirical\n')
        # iterate each row in the dataframe
        for user_row in df.values:
            # form a dataframe for the user_row
            x_df = pd.DataFrame(data=[user_row], columns=data_columns)
            robust_rectangle = robust_rect_rf(x_df)
            x_private_values = [user_row[private_ind_1], user_row[private_ind_2]]
            # write the theoretical and empirical accuracy to csv file
            for epsilon in range(1, 9):
                f.write(f'{x_private_values[0]:.2f},{x_private_values[1]:.2f}')
                f.write(f',{epsilon}')
                for mechanism in ["pm", "sw", "krr", "exp", "laplace", "gaussian"]:
                    prob_accumulated = theoretical_accuracy(x_private_values, epsilon, robust_rectangle, mechanism=mechanism)
                    accuracy = empirical_accuracy(user_row, epsilon, sample_num=3000, mechanism=mechanism)
                    f.write(f',{prob_accumulated:.6f},{accuracy:.3f}')
                f.write('\n')

