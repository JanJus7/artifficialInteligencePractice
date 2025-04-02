from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



iris = datasets.load_iris()

datasets = train_test_split(iris.data, iris.target, test_size=0.3)

train_data, test_data, train_labels, test_labels = datasets

print("Mapowanie etykiet:", {i: name for i, name in enumerate(iris.target_names)})

scaler = StandardScaler()

scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(train_data[:3])

mlp = MLPClassifier(hidden_layer_sizes=(2, ), max_iter=5000)
mlp.fit(train_data, train_labels)
predictions_test = mlp.predict(test_data)
print("Dokładność (2 neurony w warstwie ukrytej):", accuracy_score(predictions_test, test_labels))


mlp_3 = MLPClassifier(hidden_layer_sizes=(3,), max_iter=5000)
mlp_3.fit(train_data, train_labels)
predictions_test_3 = mlp_3.predict(test_data)
print("Dokładność (3 neurony w warstwie ukrytej):", accuracy_score(predictions_test_3, test_labels))

mlp_3x2 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=5000)
mlp_3x2.fit(train_data, train_labels)
predictions_test_3x2 = mlp_3x2.predict(test_data)
print("Dokładność (dwie warstwy po 3 neurony):", accuracy_score(predictions_test_3x2, test_labels))