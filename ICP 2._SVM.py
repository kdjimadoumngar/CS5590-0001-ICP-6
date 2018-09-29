

# from sklearn import svm
#
# import matplotlib as pd
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
#
#
# # Importing train_test_split to randomly split the data
#
# #reading the data
# iris = datasets.load_iris()
#
# x = iris.data
# y = iris.target
#
#
# # for labels, possible_values in labels.items():
# #     print(labels, possible_values)
#
# # Splitting the data into train and test
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
#
# pd.plot(x_train, y_train, x_test, y_test) # Plotting the train and test dataset
#
# # To create a linear classifier
#
# clf = svm.SVC(kernel='linear')
#
# clf.fit(x_train, y_train) # Training the classifier
#
# pd.plot_decision_function(x_train, y_train, x_test, y_test, clf) # Plot decision function using training and testing data
#
# # Testing the Linear SVM classifier
#
# clf_pred = clf.predict(x_test) # Making predictions using the test data
#
# print("Accuracy: {}%".format(clf.score(x_test, y_test)*100))











######################################################

# Libraries

import numpy as np

# from sklearn.model_selection import train_test_split
#
# from sklearn import datasets
#
# from sklearn import svm
#
# iris = datasets.load_iris()
#
#
# # Reading data
#
# #iris = pd.read_csv('iris.csv')
#
# iris.head()
#
#
# # Fitting the model
#
# from sklearn.svm import SVC
#
# X = iris[['sepal length', 'sepal width', 'petal length', 'petal width']]
#
# y = iris['class']
#
# X_train, X_test, y_train, y_teat = train_test_split(X, y, random_state = 0)
#
#
# svm = SVC()
#
# svm.fit(X_train, y_train)
#
#
# # Testing the Linear SVM classifier
#
# # )clf_pred = clf.predict(X_test) # Making predictions using the test data
# # #
# # # print("Accuracy: {}%".format(clf.score(X_test, y_test)*100)
#
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.30, random_state = 0)
#
# clf = svm.SVC(kernel ='linear', C = 1).fit(X_train, y_test)
#
# clf.score(X_test, y_test)

# #CV
#
# import numpy as np
#
# from sklearn.model_selection import train_test_split
#
# from sklearn import datasets
#
# from sklearn import svm
#
# iris = datasets.load_iris()
#
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.30, random_state = 0)
#
# clf = svm.SVC(kernel ='linear', C = 1).fit(X_train, y_test)
#
# clf.score(X_test, y_test)


######################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = pd.read_csv('iris.csv')
print(iris.head())
print(iris['class'].unique())
print(iris.describe())

import seaborn as sns
sns.countplot(iris['class'],label="Count")
plt.show()

from sklearn.svm import SVC

X = iris[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = iris['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svm = SVC(kernel='linear')

svm.fit(X_train, y_train)

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))


# y_pred= svm.fit(X_train, y_train).predict(X_test)

# print('Accuracy of SVM classifier on test set: {:.2f}'
#      .format(svm.score(y_pred, y_test)))
#
#
# clf_pred = clf.predict(x_test) # Making predictions using the test data

#print("Accuracy: {}%".format(clf.score(x_test, y_test)*100))
