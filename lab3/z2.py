import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df, train_size=0.7, random_state=288491)

XTrain = train_set.iloc[:, :4]
YTrain = train_set.iloc[:, 4]

XTest = test_set.iloc[:, :4]
YTest = test_set.iloc[:, 4]

print("Treningowy")
print(train_set)
print("\nTestowy")
print(test_set)

clf = tree.DecisionTreeClassifier()

clf.fit(XTrain, YTrain)

plt.figure(figsize=(20, 12))
tree.plot_tree(clf, 
          filled=True, 
          feature_names=df.columns[:4].tolist(),
          class_names=df['variety'].unique())
plt.show()

accuracy = clf.score(XTest, YTest)
print(f"\nAcc: {accuracy*100:.2f}%")

YPred = clf.predict(XTest)

errMatrix = metrics.confusion_matrix(YTest, YPred)
print(f"\n{errMatrix}")

goodPredictions = sum(YPred == YTest)
total = len(YTest)