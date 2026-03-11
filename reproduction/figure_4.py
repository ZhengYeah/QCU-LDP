import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cdf_ldp_mechanisms_at_x import CDFAtX
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter



# Enable LaTeX interpreter
plt.rcParams['font.size'] = 20

def concentration_probability(epsilon, x):
    cdf_at_x = CDFAtX(epsilon, x)
    cdf_pm = cdf_at_x._pm()
    cdf_laplace = cdf_at_x._laplace()
    cdf_sw = cdf_at_x._sw()
    cdf_exp_l1 = cdf_at_x._exp_abs()
    cdf_krr = cdf_at_x._krr()
    index_multiplier_continuous = cdf_at_x.discretization_level
    index_multiplier_discrete = cdf_at_x.bin_num

    # compute cdf concentration around x
    theta = np.linspace(0, 0.5, 50, endpoint=False)
    pm_concentration_probability = np.zeros(len(theta))
    laplace_concentration_probability = np.zeros(len(theta))
    sw_concentration_probability = np.zeros(len(theta))
    exp_concentration_probability = np.zeros(len(theta))
    exp_krr_concentration_probability = np.zeros(len(theta))
    for i in range(len(theta)):
        # continuous case
        right = int((x + theta[i]) * index_multiplier_continuous)
        left = int((x - theta[i]) * index_multiplier_continuous)
        pm_concentration_probability[i] = cdf_pm[right] - cdf_pm[left]
        laplace_concentration_probability[i] = cdf_laplace[right] - cdf_laplace[left]
        sw_concentration_probability[i] = cdf_sw[right] - cdf_sw[left]
        # discrete case
        right = int((x + theta[i]) * index_multiplier_discrete)
        left = int((x - theta[i]) * index_multiplier_discrete)
        exp_concentration_probability[i] = cdf_exp_l1[right] - cdf_exp_l1[left]
        exp_krr_concentration_probability[i] = cdf_krr[right] - cdf_krr[left]

    return theta, pm_concentration_probability, laplace_concentration_probability, sw_concentration_probability, exp_concentration_probability, exp_krr_concentration_probability

epsilon = 2
x = 0.5
theta, pm_concentration_probability, laplace_concentration_probability, sw_concentration_probability, exp_concentration_probability, exp_krr_concentration_probability = concentration_probability(epsilon, x)

# plot cdf concentration around x
plt.figure(figsize=(8, 4.8))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.plot(theta, pm_concentration_probability, 'black', label='PM', linewidth=3, linestyle='-.')
plt.plot(theta, sw_concentration_probability, 'red', label='SW', linewidth=3, linestyle='-.')
plt.plot(theta, exp_concentration_probability, 'green', label=r'Exponential', linewidth=3)
plt.plot(theta, exp_krr_concentration_probability, 'purple', label='k-RR', linewidth=3)
plt.plot(theta, laplace_concentration_probability, 'blue', label='Laplace', linewidth=3, linestyle='--')
# Remove ending zeros in the ticks
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\rho(\varepsilon, \theta)$')
plt.legend()
plt.title('Figure 4(a)')
# plt.show()

epsilon = 4
theta, pm_concentration_probability, laplace_concentration_probability, sw_concentration_probability, exp_concentration_probability, exp_krr_concentration_probability = concentration_probability(epsilon, x)

# plot cdf concentration around x
plt.figure(figsize=(8, 4.8))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.plot(theta, pm_concentration_probability, 'black', label='PM', linewidth=3, linestyle='-.')
plt.plot(theta, sw_concentration_probability, 'red', label='SW', linewidth=3, linestyle='-.')
plt.plot(theta, exp_concentration_probability, 'green', label=r'Exponential', linewidth=3)
plt.plot(theta, exp_krr_concentration_probability, 'purple', label='k-RR', linewidth=3)
plt.plot(theta, laplace_concentration_probability, 'blue', label='Laplace', linewidth=3, linestyle='--')
# Remove ending zeros in the ticks
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\rho(\varepsilon, \theta)$')
plt.legend()
plt.title('Figure 4(b)')
# plt.show()

epsilon = 6
theta, pm_concentration_probability, laplace_concentration_probability, sw_concentration_probability, exp_concentration_probability, exp_krr_concentration_probability = concentration_probability(epsilon, x)

# plot cdf concentration around x
plt.figure(figsize=(8, 4.8))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.plot(theta, pm_concentration_probability, 'black', label='PM', linewidth=3, linestyle='-.')
plt.plot(theta, sw_concentration_probability, 'red', label='SW', linewidth=3, linestyle='-.')
plt.plot(theta, exp_concentration_probability, 'green', label=r'Exponential', linewidth=3)
plt.plot(theta, exp_krr_concentration_probability, 'purple', label='k-RR', linewidth=3)
plt.plot(theta, laplace_concentration_probability, 'blue', label='Laplace', linewidth=3, linestyle='--')
# Remove ending zeros in the ticks
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\rho(\varepsilon, \theta)$')
plt.legend()
plt.title('Figure 4(c)')
plt.show()
