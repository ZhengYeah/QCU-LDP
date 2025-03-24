import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace
from matplotlib.ticker import FormatStrFormatter


# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

theta = 0.3

epsilon = np.linspace(1,9, 100)

# Parameters for the Laplace distribution
mu = 0  # Location parameter
b = np.ones(len(epsilon)) / np.array(epsilon)  # Scale parameter

# rho list
rho_list = np.zeros(len(epsilon))
for i in range(len(epsilon)):
    cdf = laplace.cdf(theta, mu, b[i])
    p = 2 * cdf - 1
    rho_list[i] = p

# Plot cdf list
plt.figure(figsize=(5, 5))
ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

plt.plot(epsilon, rho_list, 'black', linewidth=2)


# Add labels and title
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$p(0.3, \theta)$')
plt.ylim(0, 1)
plt.xlim(1, 8)

# Remove ending zeros in the ticks
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))


# Show the plot
plt.savefig('./rho_epsilon_laplace.pdf')
plt.show()
