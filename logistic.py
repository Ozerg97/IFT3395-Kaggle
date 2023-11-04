import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, Y, num_iterations, learning_rate):
    """
    Implémentation simple de la régression logistique avec descente de gradient.
    """

    n = X.shape[0]  # nombre de features
    m = X.shape[1]  # nombre de d'exemples

    # Initialiser les poids
    weights = np.zeros((n, 1))
    b = 0

    cost_list = []
    for i in range(num_iterations):
        z = np.dot(weights.T, X) + b
        predictions = sigmoid(z)

        # Cost Function
        epsilon = 0.0001
        cost = -(1 / m) * np.sum(Y * np.log(predictions + epsilon) + (1 - Y) * np.log(1 - predictions + epsilon))

        # Gradient descent
        dW = (1 / m) * np.dot(predictions - Y, X.T)
        dB = (1 / m) * np.sum(predictions - Y)

        weights = weights - learning_rate * dW.T
        b = b - learning_rate * dB

        # Keeping track of our cost function value
        cost_list.append(cost)

        if i % (num_iterations / 10) == 0:
            print("cost after ", i, "iteration is : ", cost)

    return weights, b, cost_list

def one_vs_rest_logistic_regression(X_train, Y_train, num_labels, num_iterations, learning_rate):
    """
    Former un modèle de régression logistique OvR pour la classification multi-classes.
    """
    models = []
    list_cost_list = []
    # Former un modèle pour chaque classe
    weights_list = []
    b_list = []
    for i in range(num_labels):
        # Créer des étiquettes binaires pour la classe actuelle
        binary_labels = (Y_train == i).astype(int)
        binary_labels = binary_labels.reshape(1, -1)  # Ensure the shape is (1, m)
        
        # Former un modèle de régression logistique pour la classe actuelle
        weights, b, cost_list = logistic_regression(X_train, binary_labels, num_iterations, learning_rate)
        model = {"weights": weights, "bias": b}
        models.append(model)

        '''weights_list.append(weights.T[0])
        b_list.append(b)
        list_cost_list.append(cost_list)'''

    #return weights_list, b_list, list_cost_list
    return models 

def predict_one_vs_rest(models, X):
    scores = []

    for model in models:
        weights, b = model["weights"], model["bias"]
        z = np.dot(weights.T, X) + b
        class_scores = sigmoid(z)
        scores.append(class_scores)

    predicted_labels = np.argmax(scores, axis=0)

    return predicted_labels

'''def predict_one_vs_rest(weights_list, b_list, X):
    """
    Prédire des étiquettes pour de nouvelles données en utilisant des modèles OvR.
    """

    scores = []
    # Calculer les scores pour chaque classe
    for i in range(len(weights_list)):
        z = np.dot(weights_list[i], X ) + b_list[i]
        class_scores = sigmoid(z)
        print(class_scores)
        scores.append(class_scores)

    # Retrouver l'indice de la classe avec le score le plus élevé pour chaque échantillon
    predicted_labels = np.argmax(scores, axis=0)

    return predicted_labels'''

train_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/train.csv')
test_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/test.csv')
# Séparer les caractéristiques et les étiquettes
X_train = train_data.drop(columns=["SNo", "Label"], axis=1)
Y_train = train_data["Label"]
X_test = test_data.drop("SNo", axis=1)

# Conversion en tableaux NumPy
X_train = X_train.to_numpy().T
Y_train = Y_train.to_numpy().reshape(1, -1)
X_test = X_test.to_numpy().T

'''train_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/train.csv')
test_data = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/test.csv')
X_train = train_data.drop(columns= ["SNo","Label"], axis = 1)
columns = ["SNo","lat","lon","TMQ","U850","V850","UBOT","VBOT","QREFHT","PS","PSL","T200","T500","PRECT","TS","TREFHT","Z1000","Z200","ZBOT","time",]
Y_train = train_data.drop(columns, axis = 1)
X_test = test_data.drop("SNo", axis = 1)
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
X_train = X_train.T
Y_train = Y_train.reshape(1, X_train.shape[1])
X_test = X_test.T'''


# Paramètres du modèle
num_labels = 3  # Pour les labels 0, 1, et 2
num_iterations = 500  # Doit être déterminé par la validation croisée
learning_rate = 0.1  # Doit être déterminé par la validation croisée

# Former le modèle OvR
models = one_vs_rest_logistic_regression(X_train.T, Y_train, num_labels, num_iterations, learning_rate)

# Normalisation des données de test en utilisant la moyenne et l'écart type des données d'entraînement
mean = np.mean(X_train, axis=1)
std = np.std(X_train, axis=1)
X_test_normalized = (X_test - mean[:, np.newaxis]) / std[:, np.newaxis]

# Faire des prédictions
predictions = predict_one_vs_rest(models, X_test_normalized)
print(predictions)

# Former le modèle OvR
#weights_list, b_list, list_cost_list = one_vs_rest_logistic_regression(X_train, Y_train, num_labels, num_iterations, learning_rate)


# Faire des prédictions
#predictions = predict_one_vs_rest(weights_list, b_list, X_test)