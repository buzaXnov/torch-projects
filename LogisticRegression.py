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
X_train = stand_scaler.fit_transform(X_train)
X_test = stand_scaler.transform(X_test)    # TODO: Difference between fit_transform and transform?

X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = torch.reshape(Y_train, (Y_train.shape[0], 1))
Y_test = torch.reshape(Y_test, (Y_test.shape[0], 1))

# 1) design model
class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, out_features)   # only one layer

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegression(in_features=n_features, out_features=1)  # 30x1

# 2) loss and optimizer
learning_rate = 0.01
loss = nn.BCELoss()     # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# 3) training loop
n_iters = 1000
for epoch in range(n_iters):
    # forward pass
    Y_pred = model(X_train)
    # loss calculation
    L = loss(Y_pred, Y_train)
    # backward propagation/pass
    L.backward()
    # update weights
    optimizer.step()
    # emptying the gradients
    optimizer.zero_grad()
    if not epoch % 20:
        print(f"epoch {epoch}: loss={L.item():.5f}")

with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred_classes = (Y_pred > 0.5).float()   # one-hot encoding
    # accuracy = Y_pred_classes.eq(Y_test).sum()
    accuracy = (Y_pred_classes == Y_test).float().sum() / Y_test.shape[0]
    print(f"Accuracy is {accuracy:.4f}.")
