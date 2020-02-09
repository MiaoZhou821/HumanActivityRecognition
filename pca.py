import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data_HAR = pd.read_csv('/Users/shuishui/Desktop/2019FALL/CAPSTONE/MiniProj/datasetHAR.csv')

X = data_HAR.drop('Activity', axis=1)
X = X.drop('subject', axis=1)

y = data_HAR['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
sc = StandardScaler().fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
n_com = []
n_com_acc = []
for i in range(1, 561, 10):
    data_HAR = pd.read_csv('/Users/shuishui/Desktop/2019FALL/CAPSTONE/MiniProj/datasetHAR.csv')

    X = data_HAR.drop('Activity', axis=1)
    X = X.drop('subject', axis=1)

    y = data_HAR['Activity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    sc = StandardScaler().fit(X_train)

    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    pca = PCA(n_components=i)
    pca.fit(X_train)
    n_com.append(pca.n_components_)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    svclassifier = SVC(kernel = 'linear')
    svclassifier.fit(X_train,y_train)
    y_pred = svclassifier.predict(X_test)
    n_com_acc.append(svclassifier.score(X_test,y_test))

print(n_com)
print(n_com_acc)
import pylab as pl
pl.plot(n_com, n_com_acc)
pl.xlabel('number of components')
pl.ylabel('accuracy')
pl.show()

