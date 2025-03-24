import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

# The first img (mnist_7_7_0.npy)
df = pd.read_csv('cnn_accuracy_0.csv')
plt.ylim(0, 1)
plt.xticks(range(1, 10))
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], label="PM", marker='o', color='red', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], 0.1 + 0.9 * df["pm_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_theo"], label="SW", marker='s', color='black', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_empirical"], marker='s', linestyle='--', color='black')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["sw_theo"], 0.1 + 0.9 * df["sw_empirical"], color='black', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], label="Exp", marker='x', color='green', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], 0.1 + 0.9 * df["exp_empirical"], color='green', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["krr_theo"], label="k-RR", marker='s', color='blue', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["krr_empirical"], marker='s', linestyle='--', color='blue')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["krr_theo"], 0.1 + 0.9 * df["krr_empirical"], color='blue', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=15, loc='upper left')
plt.savefig('./cnn_accuracy_0.pdf')
plt.show()


# The second img (mnist_7_7_1.npy)
df = pd.read_csv('cnn_accuracy_1.csv')
plt.ylim(0, 1)
plt.xticks(range(1,10))
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], label="PM", marker='o', color='red', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], 0.1 + 0.9 * df["pm_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_theo"], label="SW", marker='s', color='black', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["sw_empirical"], marker='s', linestyle='--', color='black')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["sw_theo"], 0.1 + 0.9 * df["sw_empirical"], color='black', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], label="Exp", marker='x', color='green', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], 0.1 + 0.9 * df["exp_empirical"], color='green', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["krr_theo"], label="k-RR", marker='s', color='blue', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["krr_empirical"], marker='s', linestyle='--', color='blue')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["krr_theo"], 0.1 + 0.9 * df["krr_empirical"], color='blue', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=15, loc='upper left')
plt.savefig('./cnn_accuracy_1.pdf')
plt.show()
