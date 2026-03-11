import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['font.size'] = 20

df = pd.read_csv('lr_avg_wor.csv')
df["epsilon"] = df["epsilon"].astype(int)
# compute the average of the theoretical and empirical accuracy for each epsilon
df["pm_theo"] = df.groupby("epsilon")["pm_theo"].transform('mean')
df["pm_empirical"] = df.groupby("epsilon")["pm_empirical"].transform('mean')
df["exp_theo"] = df.groupby("epsilon")["exp_theo"].transform('mean')
df["exp_empirical"] = df.groupby("epsilon")["exp_empirical"].transform('mean')
df["laplace_theo"] = df.groupby("epsilon")["laplace_theo"].transform('mean')
df["laplace_empirical"] = df.groupby("epsilon")["laplace_empirical"].transform('mean')
df["gaussian_theo"] = df.groupby("epsilon")["gaussian_theo"].transform('mean')
df["gaussian_empirical"] = df.groupby("epsilon")["gaussian_empirical"].transform('mean')
df = df.drop_duplicates(subset=["epsilon"])

plt.figure()
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(range(1, 9))
# shaded region between the theoretical and empirical accuracy
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
plt.ylabel(r'$\rho(\varepsilon), \hat{\rho}(\varepsilon)$')
plt.legend(fontsize=18)
plt.title('Figure 13a')


# worst-case accuracy plot
df = pd.read_csv('lr_avg_wor.csv')
df["epsilon"] = df["epsilon"].astype(int)
# compute the average of the theoretical and empirical accuracy for each epsilon
df["pm_theo"] = df.groupby("epsilon")["pm_theo"].transform('min')
df["pm_empirical"] = df.groupby("epsilon")["pm_empirical"].transform('min')
df["exp_theo"] = df.groupby("epsilon")["exp_theo"].transform('min')
df["exp_empirical"] = df.groupby("epsilon")["exp_empirical"].transform('min')
df["laplace_theo"] = df.groupby("epsilon")["laplace_theo"].transform('min')
df["laplace_empirical"] = df.groupby("epsilon")["laplace_empirical"].transform('min')
df["gaussian_theo"] = df.groupby("epsilon")["gaussian_theo"].transform('min')
df["gaussian_empirical"] = df.groupby("epsilon")["gaussian_empirical"].transform('min')
df = df.drop_duplicates(subset=["epsilon"])

plt.figure()
for spine in plt.gca().spines.values():
    spine.set_linewidth(1)
plt.ylim(0, 1)
plt.xticks(range(1, 9))
# shaded region between the theoretical and empirical accuracy
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
plt.ylabel(r'$\rho(\varepsilon), \hat{\rho}(\varepsilon)$')
plt.legend(fontsize=18)
plt.title('Figure 13b')
plt.show()