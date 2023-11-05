import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegressionMulticlass:

    def __init__(self, num_classes, lr=0.001, n_iters=1000):
        self.num_classes = num_classes
        self.lr = lr
        self.n_iters = n_iters
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.models = []

        for class_label in range(self.num_classes):
            y_class = (y == class_label).astype(int)
            model = LogisticRegression(self.lr, self.n_iters)
            model.fit(X, y_class)
            self.models.append(model)

    def predict(self, X):
        class_preds = []

        for model in self.models:
            y_pred = model.predict(X)
            class_preds.append(y_pred)

        return np.array(class_preds).T  # Transposer pour obtenir les prédictions par classe

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return y_pred

if __name__ == "__main":
    # Charger les données d'entraînement et de test (utilisez vos propres chemins de fichiers)
    train_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/train.csv')
    test_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/test.csv')

    # Séparation des caractéristiques et des étiquettes
    X_train = train_data.drop(columns=["SNo", "Label"], axis=1)
    Y_train = train_data["Label"]
    X_test = test_data.drop("SNo", axis=1)

    # Conversion en tableaux NumPy
    X_train = X_train.to_numpy().T
    Y_train = Y_train.to_numpy().reshape(1, -1)
    X_test = X_test.to_numpy().T

    num_classes = 3  # Remplacez par le nombre réel de classes dans vos données
    num_iterations = 500  # Doit être déterminé par la validation croisée
    learning_rate = 0.1  # Doit être déterminé par la validation croisée

    # Initialisez et entraînez le modèle multiclasse
    model = LogisticRegressionMulticlass(num_classes, lr=learning_rate, n_iters=num_iterations)
    model.fit(X_train, Y_train)

    # Faites des prédictions sur les données de test
    predictions = model.predict(X_test)

    print(predictions)
