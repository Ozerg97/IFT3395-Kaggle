import numpy as np
import pandas as pd
from nltk import accuracy


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #si y <= 0 -> y_ = -1 sinon 1 -> tri des classes
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

columns = ["SNo","lat","lon","TMQ","U850","V850","UBOT","VBOT","QREFHT","PS","PSL","T200","T500","PRECT","TS","TREFHT","Z1000","Z200","ZBOT","time",]

X_train = train_data.drop(columns= ["SNo","Label"], axis = 1)
X_train = X_train.values
X_train = X_train.T

X_test = test_data.drop("SNo", axis = 1)
X_test = X_test.values
X_test = X_test.T

Y_train = train_data.drop(columns, axis = 1)
Y_train = Y_train.values
Y_train = Y_train.reshape(1, X_train.shape[1])

Y_test = test_data.drop(columns, axis = 1)
Y_test = Y_test.values
Y_test = Y_test.reshape(1, X_test.shape[1])

classifier = SVM()
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)

print("SVM classification accuracy:", accuracy(Y_test, predictions))






