import matplotlib.pyplot as plt
import seaborn as sns

# Feature distributions
df.hist(figsize=(20, 15), bins=30)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.show()

# Box plots for pitch features by gender
sns.boxplot(x='label', y='mean_pitch', data=df)
plt.show()