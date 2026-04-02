import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add the parent directory to the system path to allow imports from src
BASE_DIR = Path(__file__).resolve().parent.parent # Define the base directory for the project

from src.samples_from_mechanism import samples_of_mechanism
from src.robust_radius_sklearn import RobustRadiusSKLearn
from src.cdf_ldp_mechanisms_at_x import CDFAtX

# private features: age (index 0), bmi (index 5), be cautious about the index
# choose a user from the dataset as x
user_row = [0.79,1.0,0.0,1.0,0.5804,0.48]
private_ind_1, private_ind_2 = 0, 5

private_values = [user_row[private_ind_1], user_row[private_ind_2]] # age, bmi
data_columns = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi']
x_df = pd.DataFrame(data=[user_row], columns=data_columns)
model = joblib.load(BASE_DIR / 'experiments/stroke_pred/classifiers/stroke_lr.pkl')

# time variables
empirical_sampling_time = 0
empirical_inference_time = 0
theoretical_time = 0
robust_radius_time = 0

def robust_rect_rf():
    robust_rec = RobustRadiusSKLearn(model, x_df, ['age','bmi'], 0.05, 0.03)
    start_time = time.perf_counter()
    radius = robust_rec.binary_search()
    end_time = time.perf_counter()
    robust_radius_time = end_time - start_time

    robust_rectangle = robust_rec.adjust_step_rate([(0.4, 0.5), (0.3, 0.5), (0.2, 0.5)])
    robust_rectangle = robust_rec.adjust_step_rate([(0.1, 0.5), (0.05, 0.5)])
    return robust_rectangle, robust_radius_time

def theoretical_accuracy(epsilon, robust_rectangle, mechanism="pm"):
    # compute the theoretical accuracy
    prob_accumulated = 1
    for i, private_value in enumerate(private_values):
        cdf_at_x = CDFAtX(epsilon, private_value, bin_num=100)
        rectangle = [robust_rectangle[0][i], robust_rectangle[1][i]]
        cdf_rect = cdf_at_x.cdf_of_tilde_x(rectangle, mechanism)
        prob_accumulated *= cdf_rect
    return prob_accumulated * 0.99 # term (1 - \tau) in the paper, we set tau to 0.01


def empirical_accuracy(epsilon, sample_num=3000, mechanism="pm"):
    # sampling time
    start = time.perf_counter()
    if mechanism == "laplace" or mechanism == "gaussian":
        samples, fail_num_laplace = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)
    else:
        samples = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)
    end = time.perf_counter()
    print(f"{mechanism} sampling time: {(end - start) * 1000:.3f}ms")

    # inference time
    start = time.perf_counter()
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
    if mechanism == "laplace":
        accuracy = np.sum(ground_truth == pred) / (sample_num + fail_num_laplace)
    else:
        accuracy = np.sum(ground_truth == pred) / sample_num
    end = time.perf_counter()
    print(f"{mechanism} inference time: {(end - start) * 1000:.3f}ms")
    return accuracy


if __name__ == '__main__':
    epsilon = 1
    robust_rectangle, robust_radius_time = robust_rect_rf()

    print(f"Robust radius computation time: {robust_radius_time * 1000:.3f}ms")
    print("=================================")
    print("Theoretical and empirical time comparison for epsilon")
    print("=================================")
    for mechanism in ["pm", "exp", "laplace"]:
        start = time.perf_counter()
        theoretical_accuracy(epsilon, robust_rectangle, mechanism=mechanism)
        end = time.perf_counter()
        print(f"\033[32m{mechanism} theoretical time: {(end - start) * 1000:.6f}ms\033[0m")
        # empirical time
        empirical_accuracy(epsilon, sample_num=2000, mechanism=mechanism)
