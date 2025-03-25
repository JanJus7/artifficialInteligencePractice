import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree, neighbors, naive_bayes

df = pd.read_csv('iris.csv')

def splitData(df):
    (trainSet, testSet) = train_test_split(df.values, train_size=0.7, random_state=288491)
    trainInputs = trainSet[:, 0:4]
    trainClasses = trainSet[:, 4]
    testInputs = testSet[:, 0:4]
    testClasses = testSet[:, 4]
    return trainInputs, trainClasses, testInputs, testClasses

def classifyKnn(neighborsCount, dataSet):
    trainInputs, trainClasses, testInputs, testClasses = splitData(dataSet)
    knnClassifier = neighbors.KNeighborsClassifier(n_neighbors=neighborsCount).fit(trainInputs, trainClasses)
    predictions = knnClassifier.predict(testInputs)
    accuracyScore = metrics.accuracy_score(testClasses, predictions)
    print(f"{neighborsCount} neighbors:")
    print(metrics.confusion_matrix(testClasses, predictions))
    print(f"Acc: {round(accuracyScore, 3)}\n")

def classifyDecisionTree(df):
    trainInputs, trainClasses, testInputs, testClasses = splitData(df)
    treeClassifier = tree.DecisionTreeClassifier()
    treeClassifier.fit(trainInputs, trainClasses)
    predictions = treeClassifier.predict(testInputs)
    accuracyScore = metrics.accuracy_score(testClasses, predictions)
    print("Decision Tree:")
    print(metrics.confusion_matrix(testClasses, predictions))
    print(f"Acc: {round(accuracyScore, 3)}\n")

def classifyNaiveBayes(df):
    trainInputs, trainClasses, testInputs, testClasses = splitData(df)
    nbClassifier = naive_bayes.GaussianNB().fit(trainInputs, trainClasses)
    predictions = nbClassifier.predict(testInputs)
    accuracyScore = metrics.accuracy_score(testClasses, predictions)
    print("Naive Bayes:")
    print(metrics.confusion_matrix(testClasses, predictions))
    print(f"Acc: {round(accuracyScore, 3)}\n")

classifyDecisionTree(df)
classifyKnn(3, df)
classifyKnn(5, df)
classifyKnn(11, df)
classifyNaiveBayes(df)