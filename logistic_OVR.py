import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Implémentation de la régression logistique multiclasse avec l'approche One-vs-Rest
class MultiClassLogisticRegression:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.classifiers = {}

    def fit(self, X, y):
        unique_classes = np.unique(y)

        for c in unique_classes:
            binary_y = np.where(y == c, 1, 0)
            classifier = LogisticRegression(self.learning_rate, self.num_iterations)
            classifier.fit(X, binary_y)
            self.classifiers[c] = classifier

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classifiers)))

        for c, classifier in self.classifiers.items():
            binary_pred = classifier.predict(X)
            predictions[:, c] = binary_pred

        return np.argmax(predictions, axis=1)


#Implémentation de la régression logistique binaire 
class LogisticRegression():
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.num_iterations):
            linear_pred = np.dot(X, self.w) + self.b
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.w -= (self.learning_rate * dw)
            self.b -= (self.learning_rate * db)

    def predict(self, X):
        linear_pred = np.dot(X, self.w) + self.b
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred



train_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/train.csv')
test_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/test.csv')

columns = ["SNo", "lat", "lon", "TMQ", "U850", "V850", "UBOT", "VBOT", "QREFHT", "PS", "PSL", "T200", "T500", "PRECT",
           "TS", "TREFHT", "Z1000", "Z200", "ZBOT", "time"]

# Charger les données de test
X_train = train_data.drop(columns= ["SNo","Label"], axis = 1)
X_test = test_data.drop("SNo", axis = 1)
X_train = X_train.values
X_test = X_test.values

y = train_data.drop(columns, axis=1)
y = y.values
y = y.reshape(1, X_train.shape[0])
y = y.reshape(y.shape[1], )
Y_train = y

# Calcul de la moyenne et variance de X_train
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Normalisation des données
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("Forme de X_train : ", X_train.shape)
print("Forme de Y_train : ", Y_train.shape)
print("Forme de X_test : ", X_test.shape)
print("\n")

# Entraînement
classifier = MultiClassLogisticRegression(10, 1000)
classifier.fit(X_train, Y_train)

# Prédictions
y_pred = classifier.predict(X_test)



df_predictions = pd.DataFrame({
    'SNo': range(1, len(y_pred) + 1),  # Commence à 1 et continue jusqu'à la longueur de y_pred
    'Label': y_pred
})

df_predictions.to_csv('sample_submission_logreg.csv', index=False)
