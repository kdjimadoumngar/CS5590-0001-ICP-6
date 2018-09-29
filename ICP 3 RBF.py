# Installing sklearn and matplotlib modules
# pip install -U scikit-learn
#
# pip install -U matplotlib
#
#
# import sys, os
#
# import matplotlib.pyplot as plt # iImport matplotlib.pyplot for plotting graphs
#
# from sklearm import svm # import svm from sklearn
#
# fron sklearn.model_selection import train_test_split, GridSearchCV # Importing train_test_split to randomly split the data
#
# #reading the data
#
# x, labels = read_data("iris.csv")
#
# # Splitting the data into train and test
#
# X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.30, random_state = 0)
#
# plot_data(X_train, y_train, X_test, y_test) # Plotting the train and test dataset
#
# # To create a linear classifier
#
# clf = svm.SVC(kernel='linear')
#
# clf.fit(X_train, y_train) # Training the classifier
#
# plot_decision_function(X_train, y_train, X_test, y_test, clf) # Plot decision function using training and testing data
#
# # Testing the Linear SVM classifier
#
# clf_pred = clf.predict(X_test) # Making predictions using the test data
#
# print("Accuracy: {}%".format(clf.score(X_test, y_test)*100))


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

svm = SVC(kernel='rbf')

svm.fit(X_train, y_train)

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))