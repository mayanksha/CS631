import sys
import numpy as np
import seaborn as sns
import pandas as pd

from subprocess import call
from statistics import mean
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm

def train_dtree (x_train, y_train, x_v, y_v):
    accuracies = []
    start_depth = 10
    max_depths = np.arange(start_depth, 200, 10)

    for i in max_depths:
        clf = DecisionTreeClassifier(max_depth = i)
        clf = clf.fit(x_train, y_train)

        y_pred = clf.predict(x_v)
        cm = confusion_matrix(y_v, y_pred)
        misclassification_rate = (cm[0][1] + cm[1][0]) / np.sum(cm)
        accuracies.append(1 - misclassification_rate)

    accuracies = np.array(accuracies)
    optimal_depth = np.argmax (accuracies) + 1 + start_depth
    return DecisionTreeClassifier(max_depth = optimal_depth), optimal_depth

def do_pca(x_train, y_train):
    pca = PCA(n_components=10)
    pca.fit(x_train)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('No. of components')
    plt.ylabel('Cumulative explained variance')
    plt.title ('Cumulative Explained variance vs number of components')
    plt.savefig('./variance-vs-n_components.png')

    # We need 99.99% variance to be retained, so PCA will accordingly choose minimum number of
    # components which net us aforementioned variance percentage
    pca = PCA(0.9999)
    pca.fit(x_train)
    print("Optimal number of PCA components = %d (which retain 0.9999 variance)" % (pca.n_components_))
    return pca

def do_logistic_regression(pca, x_train, y_train):
    pca_x_train = pca.transform(x_train)

    logisticRegr = LogisticRegression(solver = 'lbfgs')
    logisticRegr.fit(pca_x_train, y_train)
    return logisticRegr

def save_dtree_fig_as_png (clf):
    export_graphviz(clf, out_file='tree.dot',
                    rounded = True, proportion = False,
                    precision = 2, filled = True)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

#################### READING DATA AND SAMPLING ####################
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
###################################################################


#################### DTREE BASED CLASSIFICATION ####################
clf, optimal_depth = train_dtree (x_train, y_train, x_v, y_v)
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = 1 - (cm[0][1] + cm[1][0]) / np.sum(cm)

print("********* For Decision Tree based classifier *********")
print("Optimal Depth of Dtree (based upon validation data) = %d" % (optimal_depth))
print(cm)
print("Accuracy = %.8f" % (accuracy))
print("******************************************************")

save_dtree_fig_as_png (clf)
####################################################################


#################### PCA ####################
print("\n********* For PCA + Logistic Regression based classifier *********")
x_train_scaled = RobustScaler().fit_transform(x_train[x_train.columns])
x_v_scaled = RobustScaler().fit_transform(x_v[x_v.columns])
x_test_scaled = RobustScaler().fit_transform(x_test[x_test.columns])

pca = do_pca(x_train_scaled, y_train)
#############################################


############## PCA BASED CLASSIFICATION (LOGISTIC REGRESSION) ##############
logisticRegr = do_logistic_regression (pca, x_train_scaled, y_train)

pca_x_v = pca.transform(x_v_scaled)
pca_x_test = pca.transform(x_test_scaled)

y_pred = logisticRegr.predict(pca_x_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = 1 - (cm[0][1] + cm[1][0]) / np.sum(cm)

print(cm)
print("Accuracy = %.8f" % (accuracy))
print("******************************************************************")
############################################################################

############## PCA BASED CLASSIFICATION (USING SVM) ##############

#  f, ax = plt.subplots(figsize=(50, 50))
#  corr = x_train.corr()
#  hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f', linewidths=.05)
#  f.subplots_adjust(top=0.93)
#  plt.savefig ("./pca.png")

#  pca_x_train.hist(bins=10, color='steelblue', edgecolor='black', linewidth=0.7,
           #  xlabelsize=8, ylabelsize=8, grid=False)
#  plt.tight_layout()

#  plt.scatter(pca_x_train[:, 0], pca_x_train[:, 1],
            #  c=y_train, edgecolor='none', alpha=0.5,
            #  cmap=plt.cm.get_cmap('Spectral', 10))
#  plt.xlabel('component 1')
#  plt.ylabel('component 2')
#  plt.colorbar()

