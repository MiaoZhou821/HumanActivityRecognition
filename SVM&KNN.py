import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
#read data
data_HAR = pd.read_csv('/Users/shuishui/Desktop/2019FALL/CAPSTONE/MiniProj/datasetHAR.csv')
print(data_HAR.shape)
randomList = random.sample(range(1, 31), 6)

XY_train = data_HAR.loc[~data_HAR['subject'].isin(randomList)]

XY_test = data_HAR.loc[data_HAR['subject'].isin(randomList)]

XY_train = XY_train.drop('subject', axis=1)
X_train = XY_train.drop('Activity', axis=1)
y_train = XY_train['Activity']
XY_test = XY_test.drop('subject', axis=1)
X_test = XY_test.drop('Activity', axis=1)
y_test = XY_test['Activity']
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


labels = ["WALKING",
          "WALKING_UPSTAIRS",
          "WALKING_DOWNSTAIRS",
          "SITTING",
          "STANDING",
          "LAYING"]

#SVM Method

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train,y_train)
y_pred = svclassifier.predict(X_test)

con_svm = confusion_matrix(y_test,y_pred)
rep_svm = classification_report(y_test,y_pred)
acc_svm = svclassifier.score(X_test,y_test)
print(con_svm)
print(rep_svm)
print(acc_svm)
sns.heatmap(con_svm,square=True,annot=True,fmt ='d',cbar=False,
             xticklabels=labels,
             yticklabels=labels,cmap='Blues')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()


#KNN Method
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred3 = knn.predict(X_test)
con_knn = confusion_matrix(y_test, y_pred3)
rep_knn = classification_report(y_test,y_pred3)
acc_knn = knn.score(X_test, y_test)
print(con_knn)
print(rep_knn)
print(acc_knn)
sns.heatmap(con_knn, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=labels,
            yticklabels=labels, cmap = 'Blues')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

#vary the number of features
# from sklearn import feature_selection
# from sklearn.model_selection import cross_val_score
#
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train,y_train)
# percentiles = range(1,101,5)
# acc = []
# for i in percentiles:
#     fs = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=i)
#     X_train_fs = fs.fit_transform(X_train,y_train)
#     scores = cross_val_score(svclassifier,X_train_fs,y_train,cv=5)
#     acc = np.append(acc,scores.mean())
# print(acc)
# import pylab as pl
# pl.plot(percentiles, acc)
# pl.xlabel('percentiles of features')
# pl.ylabel('accuracy')
# pl.show()
#ACC
#[0.71525643 0.87788535 0.91452689 0.92613161 0.93483807 0.93592737
# 0.93472095 0.93641397 0.94076508 0.94221333 0.9472882  0.94825614
# 0.94982918 0.95091497 0.95780765 0.96010423 0.95889424 0.95841027
# 0.95937704 0.96361074]







