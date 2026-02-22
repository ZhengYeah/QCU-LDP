import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

# The first img (mnist_7_7_0.npy)
df = pd.read_csv('cnn_accuracy_0_monte_carlo.csv')
plt.ylim(0, 1.5)
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo_monte"], label="PM (Monte Carlo)", marker='o', linestyle='-.', color='red', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_empirical"], label="PM (Empirical)", marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["pm_theo_monte"], 0.1 + 0.9 * df["pm_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], label=r'PM ($\theta_\diamond$)', marker='o', color='red', linewidth=2.5)

plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_theo_monte"], label="SW (Monte Carlo)", marker='s', linestyle='-.', color='black', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_empirical"], label="SW (Empirical)", marker='s', linestyle='--', color='black')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["sw_theo_monte"], 0.1 + 0.9 * df["sw_empirical"], color='black', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_theo"], label=r'SW ($\theta_\diamond$)', marker='s', color='black', linewidth=2.5)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=15, loc='upper left')
plt.savefig('./cnn_accuracy_0_monte_carlo.pdf')
plt.show()


# The second img (mnist_7_7_1.npy)
df = pd.read_csv('cnn_accuracy_1_monte_carlo.csv')
plt.ylim(0, 1.5)
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo_monte"], label="PM (Monte Carlo)", marker='o', linestyle='-.', color='red', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_empirical"], label="PM (Empirical)", marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["pm_theo_monte"], 0.1 + 0.9 * df["pm_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], label=r'PM ($\theta_\diamond$)', marker='o', color='red', linewidth=2.5)

plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_theo_monte"], label="SW (Monte Carlo)", marker='s', linestyle='-.', color='black', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_empirical"], label="SW (Empirical)", marker='s', linestyle='--', color='black')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["sw_theo_monte"], 0.1 + 0.9 * df["sw_empirical"], color='black', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_theo"], label=r'SW ($\theta_\diamond$)', marker='s', color='black', linewidth=2.5)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=15, loc='upper left')
plt.savefig('./cnn_accuracy_1_monte_carlo.pdf')
plt.show()