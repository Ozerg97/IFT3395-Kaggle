import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SVM:

    def __init__(self, learning_rate, lambda_param, num_iterations):
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


# Charger les données d'entraînement
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Charger les données de test
columns = ["SNo","lat","lon","TMQ","U850","V850","UBOT","VBOT","QREFHT","PS","PSL","T200","T500","PRECT","TS","TREFHT","Z1000","Z200","ZBOT","time"]

X_train = train_data.drop(columns= ["SNo","Label"], axis = 1)
Y_train = train_data.drop(columns, axis = 1)
X_test = test_data.drop("SNo", axis = 1)
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_train = Y_train.reshape(1, X_train.shape[0])
Y_train = Y_train.reshape(Y_train.shape[1],)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = SVM(0.001, 0.01, 1000)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
predictions = classifier.predict(X_test)
df_predictions = pd.DataFrame({
    'SNo': range(1, len(y_pred) + 1),  # Commence à 1 et continue jusqu'à la longueur de y_pred
    'Label': y_pred
})

# Enregistrer les prédictions dans un fichier CSV
df_predictions.to_csv('sample_submission.csv', index=False)






