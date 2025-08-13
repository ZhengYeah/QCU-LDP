from src.samples_from_mechanism import samples_of_mechanism
from src.robust_radius_sklearn import RobustRadiusSKLearn
from src.cdf_ldp_mechanisms_at_x import CDFAtX
import numpy as np
import pandas as pd
import joblib

# private features: age (index 0), bmi (index 5), be cautious about the index
private_ind_1, private_ind_2 = 0, 5
data_columns = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi']
model = joblib.load('classifiers/stroke_lr.pkl')

def robust_rect_rf(x_dataframe):
    # note: x_dataframe = pd.DataFrame(data=[user_row], columns=data_columns)
    robust_rec = RobustRadiusSKLearn(model, x_dataframe, ['age','bmi'], 0.05, 0.02)
    radius = robust_rec.binary_search()
    robust_rectangle = robust_rec.adjust_step_rate([(0.4, 0.5), (0.3, 0.5), (0.2, 0.5)])
    robust_rectangle = robust_rec.adjust_step_rate([(0.1, 0.5), (0.05, 0.5)])
    return robust_rectangle

def theoretical_accuracy(private_values, epsilon, robust_rectangle, mechanism="pm"):
    # only need the private values in the user_row
    # compute the theoretical accuracy
    prob_accumulated = 1
    for i, private_value in enumerate(private_values):
        cdf_at_x = CDFAtX(epsilon, private_value, bin_num=100)
        rectangle = [robust_rectangle[0][i], robust_rectangle[1][i]]
        cdf_rect = cdf_at_x.cdf_of_tilde_x(rectangle, mechanism)
        prob_accumulated *= cdf_rect
    return prob_accumulated

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
    df = pd.read_csv('classifiers/processed_stroke_data_normalized.csv')
    df = df.iloc[:100].drop(columns=['stroke'])
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
