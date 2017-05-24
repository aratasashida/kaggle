# coding: utf-8
# 各ライブラリをインポート
from sklearn import datasets
import numpy as np
import pandas as pd
from scipy.spatial import distance

# irisデータセットを取得
iris = datasets.load_iris()
x = iris.data
y = iris.target

# ユークリッド距離
def euc(a,b):
    return distance.euclidean(a,b)

# K近傍法を定義
class ScrappyKNN():
    # fit method
    def fit(self,x_train,y_train):
        self.X_train = x_train
        self.Y_train = y_train

    # predict method
    def predict(self,X_test):
        predictions = []
        for row in X_test:
             label = self.closest(row)
             predictions.append(label)
        return predictions

    # closest method
    def closest(self,row):
        best_dist = euc(row,self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row,self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]

# 交差検定
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .5)

# 学習
my_classifier = ScrappyKNN()
my_classifier.fit(x_train,y_train)
predictions = my_classifier.predict(x_test)

# 精度
from sklearn.metrics import accuracy_score
print(accuracy_score(predictions,y_test))
