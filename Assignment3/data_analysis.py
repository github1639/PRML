import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.makedirs("figures", exist_ok=True)

sns.set(style="whitegrid")

df = pd.read_csv("LSTM-Multivariate_pollution.csv")
df = df[['date', 'pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']]
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()

#热力图（相关性矩阵）
plt.figure(figsize=(10, 8))
corr = df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("figures/heatmap_correlation.png")
plt.close()

#散点图矩阵（pairplot）
sns.pairplot(df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']])
plt.suptitle("Scatter Matrix of Variables", y=1.02)
plt.savefig("figures/scatter_matrix.png")
plt.close()
