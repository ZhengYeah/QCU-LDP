import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sympy.printing.pretty.pretty_symbology import line_width

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

# Random Forest comparison
df = pd.read_csv('rf_accuracy.csv')
plt.ylim(0, 1)
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], label="PM", marker='o', color='red', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], 0.1 + 0.9 * df["pm_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], label="Exp", marker='x', color='green', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], 0.1 + 0.9 * df["exp_empirical"], color='green', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["laplace_theo"], label="Laplace", marker='s', color='blue', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["laplace_empirical"], marker='s', linestyle='--', color='blue')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["laplace_theo"], 0.1 + 0.9 * df["laplace_empirical"], color='blue', alpha=0.08)
plt.plot(df["epsilon"], df["gaussian_theo"], label="Gaussian", marker='^', color='purple', linewidth=2.5)
plt.plot(df["epsilon"], df["gaussian_empirical"], marker='^', linestyle='--', color='purple')
plt.fill_between(df["epsilon"], df["gaussian_theo"], df["gaussian_empirical"], color='purple', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=18)
plt.savefig('./rf_accuracy.pdf')
plt.show()

# Logistic Regression comparison
df = pd.read_csv('lr_accuracy.csv')
plt.ylim(0, 1)
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], label="PM", marker='o', color='red', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["pm_empirical"], marker='o', linestyle='--', color='red')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["pm_theo"], 0.1 + 0.9 * df["pm_empirical"], color='red', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], label="Exp", marker='x', color='green', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["exp_theo"], 0.1 + 0.9 * df["exp_empirical"], color='green', alpha=0.08)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["laplace_theo"], label="Laplace", marker='s', color='blue', linewidth=2.5)
plt.plot(df["epsilon"], 0.1 + 0.9 * df["laplace_empirical"], marker='s', linestyle='--', color='blue')
plt.fill_between(df["epsilon"], 0.1 + 0.9 * df["laplace_theo"], 0.1 + 0.9 * df["laplace_empirical"], color='blue', alpha=0.08)
plt.plot(df["epsilon"], df["gaussian_theo"], label="Gaussian", marker='^', color='purple', linewidth=2.5)
plt.plot(df["epsilon"], df["gaussian_empirical"], marker='^', linestyle='--', color='purple')
plt.fill_between(df["epsilon"], df["gaussian_theo"], df["gaussian_empirical"], color='purple', alpha=0.08)
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend(fontsize=18)
plt.savefig('./lr_accuracy.pdf')
plt.show()