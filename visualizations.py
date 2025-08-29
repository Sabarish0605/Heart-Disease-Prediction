import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv("C:\\Users\\sabar\\Desktop\\Disease predicition\\heart.csv")  # Update with the correct file name

# Histogram for Age
plt.figure(figsize=(10, 6))
plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Boxplot for Cholesterol Levels
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['cholesterol'], color='lightgreen')
plt.title('Cholesterol Levels Boxplot')
plt.xlabel('Cholesterol')
plt.grid(axis='x', alpha=0.75)
plt.show()

# Heatmap for Correlations
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()