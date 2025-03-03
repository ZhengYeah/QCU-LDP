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
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], df["pm_theo"], label="PM", marker='o', color='red')
plt.plot(df["epsilon"], df["pm_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], df["pm_empirical"] - df["pm_std"], df["pm_empirical"] + df["pm_std"], color='red', alpha=0.2)
plt.plot(df["epsilon"], df["sw_theo"], label="SW", marker='s', color='black')
plt.plot(df["epsilon"], df["sw_empirical"], marker='s', linestyle='--', color='black')
plt.fill_between(df["epsilon"], df["sw_empirical"] - df["sw_std"], df["sw_empirical"] + df["sw_std"], color='black', alpha=0.2)
plt.plot(df["epsilon"], df["exp_theo"], label="Exp", marker='x', color='green')
plt.plot(df["epsilon"], df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], df["exp_empirical"] - df["exp_std"], df["exp_empirical"] + df["exp_std"], color='green', alpha=0.2)
plt.plot(df["epsilon"], df["krr_theo"], label="k-RR", marker='s', color='blue')
plt.plot(df["epsilon"], df["krr_empirical"], marker='s', linestyle='--', color='blue')
plt.fill_between(df["epsilon"], df["krr_empirical"] - df["krr_std"], df["krr_empirical"] + df["krr_std"], color='blue', alpha=0.2)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=15, loc='upper left')
plt.savefig('./cnn_accuracy_0.pdf')
plt.show()


# The second img (mnist_7_7_1.npy)
df = pd.read_csv('cnn_accuracy_1.csv')
plt.ylim(0, 1)
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], df["pm_theo"], label="PM", marker='o', color='red')
plt.plot(df["epsilon"], df["pm_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], df["pm_empirical"] - df["pm_std"], df["pm_empirical"] + df["pm_std"], color='red', alpha=0.2)
plt.plot(df["epsilon"], df["sw_theo"], label="SW", marker='s', color='black')
plt.plot(df["epsilon"], df["sw_empirical"], marker='s', linestyle='--', color='black')
plt.fill_between(df["epsilon"], df["sw_empirical"] - df["sw_std"], df["sw_empirical"] + df["sw_std"], color='black', alpha=0.2)
plt.plot(df["epsilon"], df["exp_theo"], label="Exp", marker='x', color='green')
plt.plot(df["epsilon"], df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], df["exp_empirical"] - df["exp_std"], df["exp_empirical"] + df["exp_std"], color='green', alpha=0.2)
plt.plot(df["epsilon"], df["krr_theo"], label="k-RR", marker='s', color='blue')
plt.plot(df["epsilon"], df["krr_empirical"], marker='s', linestyle='--', color='blue')
plt.fill_between(df["epsilon"], df["krr_empirical"] - df["krr_std"], df["krr_empirical"] + df["krr_std"], color='blue', alpha=0.2)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=15, loc='upper left')
plt.savefig('./cnn_accuracy_1.pdf')
plt.show()
