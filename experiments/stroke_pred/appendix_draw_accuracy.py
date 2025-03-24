import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

# Random Forest comparison
df = pd.read_csv('appendix_rf_accuracy.csv')
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(range(1, 9))
# shaded region between the theoretical and empirical accuracy
plt.plot(df["epsilon"], df["sw_theo"], label="SW", marker='o', color='red', linewidth=2.5)
plt.plot(df["epsilon"], df["sw_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], df["sw_theo"], df["sw_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], df["krr_theo"], label="k-RR", marker='x', color='green', linewidth=2.5)
plt.plot(df["epsilon"], df["krr_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], df["krr_theo"], df["krr_empirical"], color='green', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=18)
plt.savefig('./appendix_rf_accuracy.pdf')
plt.show()

# Logistic Regression comparison
df = pd.read_csv('appendix_lr_accuracy.csv')
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(range(1, 9))
# shaded region between the theoretical and empirical accuracy
plt.plot(df["epsilon"], df["sw_theo"], label="SW", marker='o', color='red', linewidth=2.5)
plt.plot(df["epsilon"], df["sw_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], df["sw_theo"], df["sw_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], df["krr_theo"], label="k-RR", marker='x', color='green', linewidth=2.5)
plt.plot(df["epsilon"], df["krr_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], df["krr_theo"], df["krr_empirical"], color='green', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=18)
plt.savefig('./appendix_lr_accuracy.pdf')
plt.show()