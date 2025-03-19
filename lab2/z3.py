import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

irisDf = pd.read_csv('iris.csv')

X = irisDf[['sepal.length', 'sepal.width']]
y = irisDf['variety']

plt.figure(figsize=(8, 6))
for species in y.unique():
    plt.scatter(X[y == species]['sepal.length'], X[y == species]['sepal.width'], label=species)
plt.title('Oryginalne dane')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()

minMaxScaler = MinMaxScaler()
XMinmax = minMaxScaler.fit_transform(X)
XMinmaxDf = pd.DataFrame(XMinmax, columns=['sepal.length', 'sepal.width'])

plt.figure(figsize=(8, 6))
for species in y.unique():
    plt.scatter(XMinmaxDf[y == species]['sepal.length'], XMinmaxDf[y == species]['sepal.width'], label=species)
plt.title('Znormalizowane min-max')
plt.xlabel('Sepal Length (min-max)')
plt.ylabel('Sepal Width (min-max)')
plt.legend()
plt.show()

standardScaler = StandardScaler()
XZscore = standardScaler.fit_transform(X)
XZscoreDf = pd.DataFrame(XZscore, columns=['sepal.length', 'sepal.width'])

plt.figure(figsize=(8, 6))
for species in y.unique():
    plt.scatter(XZscoreDf[y == species]['sepal.length'], XZscoreDf[y == species]['sepal.width'], label=species)
plt.title('Zeskalowane z-score')
plt.xlabel('Sepal Length (z-score)')
plt.ylabel('Sepal Width (z-score)')
plt.legend()
plt.show()
