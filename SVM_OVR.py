import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MultiClassSVM:

    def __init__(self, learning_rate, lambda_param, num_iterations):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.classifiers = {}

    def fit(self, X, y):
        unique_classes = np.unique(y)

        for c in unique_classes:
            binary_y = np.where(y == c, 1, -1)
            classifier = SVM(self.learning_rate, self.lambda_param, self.num_iterations)
            classifier.fit(X, binary_y)
            self.classifiers[c] = classifier

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))

        for c, classifier in self.classifiers.items():
            binary_pred = classifier.predict(X)
            predictions[:, c] = binary_pred

        return np.argmax(predictions, axis=1)


class SVM:

    def __init__(self, learning_rate, lambda_param, num_iterations):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# Charger les données d'entraînement
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Charger les données de test
X = train_data.drop(columns=["SNo", "Label"], axis=1)
columns = ["SNo", "lat", "lon", "TMQ", "U850", "V850", "UBOT", "VBOT", "QREFHT", "PS", "PSL", "T200", "T500", "PRECT",
           "TS", "TREFHT", "Z1000", "Z200", "ZBOT", "time"]
y = train_data.drop(columns, axis=1)

X = X.values
y = y.values
y = y.reshape(1, X.shape[0])
y = y.reshape(y.shape[1], )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement
classifier = MultiClassSVM(0.001, 0.01, 1000)
classifier.fit(X_train, y_train)

# Prédictions
y_pred = classifier.predict(X_test)

df_predictions = pd.DataFrame({
    'SNo': range(1, len(y_pred) + 1),  # Commence à 1 et continue jusqu'à la longueur de y_pred
    'Label': y_pred
})

df_predictions.to_csv('sample_submission.csv', index=False)

print(accuracy_score(y_test, y_pred))
