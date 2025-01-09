import numpy as np
from src.cdf_ldp_mechanisms_at_x import CDFAtX
import matplotlib.pyplot as plt

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20


epsilon = 2
x = 0.5

cdf_at_x = CDFAtX(epsilon, x)
cdf_pm = cdf_at_x.pm()
cdf_laplace = cdf_at_x.laplace()
cdf_sw = cdf_at_x.sw()
cdf_exp_l1 = cdf_at_x.exp_l1()
cdf_exp_l2 = cdf_at_x.exp_l2()
bin_num = cdf_at_x.discretization_level

# compute cdf concentration around x
theta = np.linspace(0, 0.5, 50, endpoint=False)
pm_concentration_probability = np.zeros(len(theta))
laplace_concentration_probability = np.zeros(len(theta))
sw_concentration_probability = np.zeros(len(theta))
exp_l1_concentration_probability = np.zeros(len(theta))
exp_l2_concentration_probability = np.zeros(len(theta))
for i in range(len(theta)):
    right = int((x + theta[i]) * bin_num)
    left = int((x - theta[i]) * bin_num)
    pm_concentration_probability[i] = cdf_pm[right] - cdf_pm[left]
    laplace_concentration_probability[i] = cdf_laplace[right] - cdf_laplace[left]
    sw_concentration_probability[i] = cdf_sw[right] - cdf_sw[left]
    exp_l1_concentration_probability[i] = cdf_exp_l1[right] - cdf_exp_l1[left]
    exp_l2_concentration_probability[i] = cdf_exp_l2[right] - cdf_exp_l2[left]

# plot cdf concentration around x
plt.figure(figsize=(10, 6))
ax = plt.gca()
plt.plot(theta, pm_concentration_probability, 'black', label='Piecewise Mechanism', linewidth=2)
plt.plot(theta, laplace_concentration_probability, 'blue', label='Laplace Mechanism', linewidth=2)
plt.plot(theta, sw_concentration_probability, 'red', label='Square Wave Mechanism', linewidth=2)
plt.plot(theta, exp_l1_concentration_probability, 'green', label='Exponential Mechanism (L1)', linewidth=2)
plt.plot(theta, exp_l2_concentration_probability, 'purple', label='Exponential Mechanism (L2)', linewidth=2)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\varepsilon, \theta)$')
plt.legend()
plt.savefig('./utility_bound_epsilon_2.pdf')
plt.show()

