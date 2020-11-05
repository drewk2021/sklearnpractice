import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


X, y = load_breast_cancer(return_X_y=True)
print(load_breast_cancer().DESCR)

training_accuracy =[]
test_accuracy = []
numbins = range(2,15)
"""
for i in numbins:
    transformer = KBinsDiscretizer(n_bins = i, encode = "onehot", strategy = "uniform") # equal-frequency
    X_binned = transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_binned, y, random_state=0)


    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(numbins, training_accuracy, label='Accuracy of the Training Set')
plt.plot(numbins, test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Bins in Discretization')
plt.legend()
plt.show()
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(f"Accuracy{clf.score(X_test,y_test)}")


importance = clf.coef_
# plot feature importance
plt.bar([x for x in range(len(importance[0]))], importance[0])
plt.show()



normalizeScaler = Normalizer(norm="l2")
X_normalized = normalizeScaler.transform(X)
for i in numbins:
    transformer = KBinsDiscretizer(n_bins = i, encode = "onehot", strategy = "uniform") # equal-frequency
    X_binned = transformer.fit_transform(X_normalized)

    X_train, X_test, y_train, y_test = train_test_split(X_binned, y, random_state=0)


    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(numbins, training_accuracy, label='Accuracy of the Training Set')
plt.plot(numbins, test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Bins in Discretization')
plt.legend()
plt.show()


transformer = KBinsDiscretizer(n_bins = 5, encode = "onehot", strategy = "uniform") # equal-frequency
X_binned = transformer.fit_transform(X_normalized)

X_train, X_test, y_train, y_test = train_test_split(X_binned, y, random_state=0)
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(f"Accuracy{clf.score(X_test,y_test)}")

importance = clf.coef_
# plot feature importance
plt.bar([x for x in range(len(importance[0]))], importance[0])
plt.show()








"""
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

transformer = KBinsDiscretizer(n_bins = 5, encode = "onehot", strategy = "uniform") # equal-frequency
X_binned = transformer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_binned, y, random_state=0)

forest.fit(X_train, y_train)
print(f"Accuracy:{forest.score(X_test,y_test)}")

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the tree
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
"""
