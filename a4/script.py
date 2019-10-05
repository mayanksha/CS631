import sys
import numpy as np
import seaborn as sns
import pandas as pd

from statistics import mean
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

a = pd.read_excel('./HW_TESLA.xlt')
cols_to_choose = a.columns[1:]

rows = len(a)
train_set = a.sample(n=rows//2)
x_train = train_set[cols_to_choose]
y_train = train_set[a.columns[0]]

remain_set = a.copy().drop(train_set.index)
validation_set = remain_set.sample(n=len(remain_set)//2)
x_v = validation_set[cols_to_choose]
y_v = validation_set[a.columns[0]]

test_set = remain_set.drop(validation_set.index)
x_test = test_set[cols_to_choose]
y_test = test_set[test_set.columns[0]]

MRs = []
max_depths = np.arange(1, 40)

for i in max_depths:
    print("Training for max_depth = %d" % (i))
    clf = DecisionTreeClassifier(max_depth = i)
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_v)
    cm = confusion_matrix(y_v, y_pred)
    misclassification_rate = (cm[0][1] + cm[1][0]) / np.sum(cm)
    MRs.append(misclassification_rate)
    print(cm)
    #  print(classification_report(y_v, y_pred))

#  plt.scatter(max_depths, MRs)
plt.plot(max_depths, MRs)
plt.show()
