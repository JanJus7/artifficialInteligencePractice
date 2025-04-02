from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv("diabetes.csv")
X = data.drop("class", axis=1)
y = data["class"].map(
    {"tested_negative": 0, "tested_positive": 1}
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation="relu", max_iter=500)
mlp.fit(X_train, y_train)

mlp1 = MLPClassifier(hidden_layer_sizes=(10, 5), activation="relu", max_iter=500)
mlp1.fit(X_train, y_train)

mlp2 = MLPClassifier(hidden_layer_sizes=(6, 4, 2), activation="relu", max_iter=500)
mlp2.fit(X_train, y_train)

mlp3 = MLPClassifier(hidden_layer_sizes=(6, 3), activation="tanh", max_iter=500)
mlp3.fit(X_train, y_train)

mlp4 = MLPClassifier(hidden_layer_sizes=(6, 3), activation="logistic", max_iter=500)
mlp4.fit(X_train, y_train)


predictions = mlp.predict(X_test)
print("Dokładność:", accuracy_score(y_test, predictions))
print("Macierz błędów:\n", confusion_matrix(y_test, predictions))

predictions1 = mlp1.predict(X_test)
print("Dokładność:", accuracy_score(y_test, predictions1))
print("Macierz błędów:\n", confusion_matrix(y_test, predictions1))

predictions2 = mlp2.predict(X_test)
print("Dokładność:", accuracy_score(y_test, predictions2))
print("Macierz błędów:\n", confusion_matrix(y_test, predictions2))

predictions3 = mlp3.predict(X_test)
print("Dokładność:", accuracy_score(y_test, predictions3))
print("Macierz błędów:\n", confusion_matrix(y_test, predictions3))

predictions4 = mlp4.predict(X_test)
print("Dokładność:", accuracy_score(y_test, predictions4))
print("Macierz błędów:\n", confusion_matrix(y_test, predictions4))

# nie poradziła sobie lepiej
# FN są gorsze bo mogą przewidzieć że nie ma cukrzycy pomimo że faktycznie jest osoba na nią chora
