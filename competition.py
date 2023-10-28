import numpy as np
import pandas as pd
import scipy.sparse as sp
#a effacer 
import matplotlib.pyplot as plt


#remplacer par le chemin d'acces du train.csv et test.csv
train = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/train.csv').values
test = pd.read_csv('/Users/christieembeya/Documents/Automne 2023/IFT 3395 ML/competition kaggle/classification-of-extreme-weather-events-udem/test.csv').values
train_label = np.array([ligne[-1] for ligne in train])
w0 = np.random.uniform(0, 1, 16)#le poids du modele
trans = np.transpose(train) 
X=[]
for i in range(len(trans)):
    if i >= 3 and i <=18:
        X.append(trans[i])
X_train = np.transpose(X) 


transs = np.transpose(train) 
y=[]
for i in range(len(transs)):
    if i >= 3 and i <=18:
        y.append(trans[i])
X_test = np.transpose(y) 

class LogisticRegression:
    def __init__(self, w0, reg):
        self.w = np.array(w0, dtype=float)
        self.reg = reg

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def loss(self, X, y):
        m = len(y)
        h = self.sigmoid(np.dot(X, self.w))
        J = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        reg_term = 0.5 * self.reg * np.sum(self.w**2)
        return J + reg_term

    def gradient(self, X, y):
        m = len(y)
        h = self.sigmoid(np.dot(X, self.w))
        error = h - y
        grad = np.dot(X.T, error) / m
        reg_term = self.reg * self.w
        return grad + reg_term

    def train(self, X, y, stepsize, num_iterations):
        for _ in range(num_iterations):
            grad = self.gradient(X, y)
            self.w -= stepsize * grad

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.w)) >= 0.5).astype(int)

# Initialisez les hyperparamètres ici (par exemple, reg, stepsize)
reg = 0.01
stepsize = 0.1
num_iterations = 1000
# Créez une instance de la régression logistique
logistic_regression = LogisticRegression(w0, reg)

# Entraînez le modèle
logistic_regression.train(X_train, train_label, stepsize, num_iterations)

# Faites des prédictions sur l'ensemble de test
predictions = logistic_regression.predict(X_test)
print(predictions)

'''feature_subset = [0,1,2]
train_label = np.array([ligne[-1] for ligne in train])
w0 = np.random.uniform(1, 10, 20)

# on ajoute une colonne pour le biais
new_train = np.insert(train, -1, 1, axis=1)
new_test = np.insert(test, -1, 1, axis=1)

#la fonction sigmoid sera utile pour la regression logistique
def sigmoid(X):
    g = 1 / (1 + np.exp(-X))
    return g


class LinearModel:
    """"Classe parent pour tous les modèles linéaires.
    """

    def __init__(self, w0, reg):
        """Les poids et les biais sont définis dans w.
        L'hyperparamètre de régularisation est reg.
        """
        self.w = np.array(w0, dtype=float)
        self.reg = reg

    def predict(self, X):
        """Retourne f(x) pour un batch X
        """
        return np.dot(X, self.w)

    def error_rate(self, X, y):
        """Retourne le taux d'erreur pour un batch X
        """
        return np.mean(self.predict(X) * y < 0)

    # les méthodes loss et gradient seront redéfinies dans les classes enfants
    def loss(self, X, y):
        return 0

    def gradient(self, X, y):
        return self.w

    def train(self, data, stepsize, n_steps, plot=False):
        """Faire la descente du gradient avec batch complet pour n_steps itération
        et un taux d'apprentissage fixe. Retourne les tableaux de loss et de
        taux d'erreur vu apres chaque iteration.
        """

        X = data[:,:-1]
        y = data[:,-1]
        losses = []
        errors = []

        for i in range(n_steps):
            # Gradient Descent
            self.w -= stepsize * self.gradient(X, y)

            # Update losses
            losses += [self.loss(X, y)]

            # Update errors
            errors += [self.error_rate(X, y)]



        print("Training completed: the train error is {:.2f}%".format(errors[-1]*100))
        return np.array(losses), np.array(errors)

def test_model(modelclass, w0=[-3.0, 3.0, 0.1], reg=.1, stepsize=.2, plot=False):
    """Crée une instance de modelclass, entraîne la, calcule le taux d'erreurs sur un
    test set, trace les courbes d'apprentissage et la frontieres de decision.
    """
    model = modelclass(w0, reg)
    training_loss, training_error = model.train(train, stepsize, 100, plot=plot)
    print("The test error is {:.2f}%".format(
      model.error_rate(test[:,:-1], test[:,-1])*100))
    print('Initial weights: ', w0)
    print('Final weights: ', model.w)

    # learning curves
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8,2))
    ax0.plot(training_loss)
    ax0.set_title('loss')
    ax1.plot(training_error)
    ax1.set_title('error rate')


class LogisticRegression(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        return np.mean(np.log(1 + np.exp(-y * self.predict(X)))) + .5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        probas = 1 / (1 + np.exp(y * self.predict(X)))
        return ((probas * -y)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w


test_model(LogisticRegression, plot=True)'''