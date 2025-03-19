from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
# print(X.head())

pca_iris = PCA(n_components=3).fit(iris.data)
print(pca_iris)
print("Wyjaśnione proporcje wariancji:", pca_iris.explained_variance_ratio_)
print("Składowe główne:", pca_iris.components_)
X_pca = pca_iris.transform(iris.data)
print("Przekształcone dane:", X_pca)

cumulative_variance = pca_iris.explained_variance_ratio_.cumsum()
n_components = (cumulative_variance < 0.95).sum() + 1
print("Liczba komponentów do zachowania 95% wariancji:", n_components)

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
target_names = iris.target_names

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA zbioru Iris')
plt.xlabel('Pierwszy komponent główny')
plt.ylabel('Drugi komponent główny')
plt.show()