import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
X, Y = datasets.load_breast_cancer(return_X_y=True)
n_samples, n_features = X.shape

# Above is the shorter way of doing this.
# binary_clf = datasets.load_breast_cancer()
# X1, Y1 = binary_clf.data, binary_clf.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# scaling our features
stand_scaler = StandardScaler()  # our features will have 0 mean and UNIT(?) variance
