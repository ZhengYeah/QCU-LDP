import numpy as np
from scipy.stats import laplace

exp = np.e

theta = 0.1
epsilon = 2

# Laplace
cdf_laplace = laplace.cdf(theta, 0, 1 / epsilon)

# Piecewise-based
C = (exp ** (epsilon / 2) - 1) / (exp ** epsilon - 1)
# assert 2 * theta <= C
cdf_piecewise = 2 * theta * exp ** (epsilon / 2)

# Exponential
x = 0.5
tilde_x = np.linspace(0, 1, 256, endpoint=True)
denominator_list = np.zeros(len(tilde_x))
for i in range(len(tilde_x)):
    denominator_list[i] = exp ** (epsilon * -abs(tilde_x[i] - x))
denominator = np.sum(denominator_list)
prob_list = np.zeros(len(tilde_x))
for i in range(len(tilde_x)):
    prob_list[i] = exp ** (epsilon * -abs(tilde_x[i] - x)) / denominator
cdf_exponential = 0
for i in range(len(tilde_x)):
    if x - theta <= tilde_x[i] <= x + theta:
        cdf_exponential += prob_list[i]

# Print results
print('Laplace:', cdf_laplace)
print('Piecewise-based:', cdf_piecewise)
print('Exponential:', cdf_exponential)
