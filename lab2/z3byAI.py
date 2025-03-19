import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')

sepal_length = X['sepal length (cm)']
sepal_width = X['sepal width (cm)']

plt.figure(figsize=(8, 6))
for i, target_name in zip([0, 1, 2], iris.target_names):
    plt.scatter(sepal_length[y == i], sepal_width[y == i], label=target_name)
plt.title('Oryginalne dane')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()

scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
sepal_length_minmax = X_minmax[:, 0]
sepal_width_minmax = X_minmax[:, 1]

plt.figure(figsize=(8, 6))
for i, target_name in zip([0, 1, 2], iris.target_names):
    plt.scatter(sepal_length_minmax[y == i], sepal_width_minmax[y == i], label=target_name)
plt.title('Znormalizowane min-max')
plt.xlabel('Sepal Length (min-max)')
plt.ylabel('Sepal Width (min-max)')
plt.legend()
plt.show()

scaler_zscore = StandardScaler()
X_zscore = scaler_zscore.fit_transform(X)
sepal_length_zscore = X_zscore[:, 0]
sepal_width_zscore = X_zscore[:, 1]

plt.figure(figsize=(8, 6))
for i, target_name in zip([0, 1, 2], iris.target_names):
    plt.scatter(sepal_length_zscore[y == i], sepal_width_zscore[y == i], label=target_name)
plt.title('Zeskalowane z-score')
plt.xlabel('Sepal Length (z-score)')
plt.ylabel('Sepal Width (z-score)')
plt.legend()
plt.show()

print("Statystyki dla oryginalnych danych:")
print("Min:", sepal_length.min(), sepal_width.min())
print("Max:", sepal_length.max(), sepal_width.max())
print("Mean:", sepal_length.mean(), sepal_width.mean())
print("Standard Deviation:", sepal_length.std(), sepal_width.std())

print("\nStatystyki dla danych znormalizowanych min-max:")
print("Min:", sepal_length_minmax.min(), sepal_width_minmax.min())
print("Max:", sepal_length_minmax.max(), sepal_width_minmax.max())
print("Mean:", sepal_length_minmax.mean(), sepal_width_minmax.mean())
print("Standard Deviation:", sepal_length_minmax.std(), sepal_width_minmax.std())

print("\nStatystyki dla danych zeskalowanych z-score:")
print("Min:", sepal_length_zscore.min(), sepal_width_zscore.min())
print("Max:", sepal_length_zscore.max(), sepal_width_zscore.max())
print("Mean:", sepal_length_zscore.mean(), sepal_width_zscore.mean())
print("Standard Deviation:", sepal_length_zscore.std(), sepal_width_zscore.std())