import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Enable LaTeX interpreter
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

# Random Forest comparison
df = pd.read_csv('rf_accuracy.csv')
plt.ylim(0, 1)
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], df["pm_theo"], label="PM", marker='o', color='red')
plt.plot(df["epsilon"], df["pm_empirical"], marker='o', linestyle='--', color='red')
# plt.plot(df["epsilon"], df["sw_theo"], label="SW", marker='s')
# plt.plot(df["epsilon"], df["sw_empirical"], marker='s', linestyle='--')
plt.plot(df["epsilon"], df["exp_theo"], label="Exp", marker='x', color='green')
plt.plot(df["epsilon"], df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.plot(df["epsilon"], df["laplace_theo"], label="Laplace", marker='s', color='blue')
plt.plot(df["epsilon"], df["laplace_empirical"], marker='s', linestyle='--', color='blue')
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend()
plt.savefig('./rf_accuracy.pdf')
plt.show()

# Logistic Regression comparison
df = pd.read_csv('lr_accuracy.csv')
plt.ylim(0, 1)
plt.xticks(range(1, 9))
plt.plot(df["epsilon"], df["pm_theo"], label="PM", marker='o', color='red')
plt.plot(df["epsilon"], df["pm_empirical"], marker='o', linestyle='--', color='red')
plt.plot(df["epsilon"], df["exp_theo"], label="Exp", marker='x', color='green')
plt.plot(df["epsilon"], df["exp_empirical"], marker='x', linestyle='--', color='green')
plt.plot(df["epsilon"], df["laplace_theo"], label="Laplace", marker='s', color='blue')
plt.plot(df["epsilon"], df["laplace_empirical"], marker='s', linestyle='--', color='blue')
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r'$\rho(\varepsilon)$')
plt.legend()
plt.savefig('./lr_accuracy.pdf')
plt.show()