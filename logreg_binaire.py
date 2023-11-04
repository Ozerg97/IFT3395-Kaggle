import numpy as np


class MyLogisticRegression:

    # taux_apprentissage: vitesse d'apprentissage du modèle
    # nbr_iterations: nombre de fois que l'algo de gradient est appelé
    def __init__(self, learning_rate=1, num_iterations=2000):
        self.taux_apprentissage = learning_rate
        self.nbr_iterations = num_iterations
        self.w = []
        self.b = 0

    def initialize_weight(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        Argument:
        dim -- size of the w vector we want (or number of parameters  in this case)
        """

        w = np.zeros((dim, 1))
        b = 0
        return w, b

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Argument:
        z -- is the decision boundary of the classifier
        """
        s = 1 / (1 + np.exp(-z))
        return s

    def hypothesis(self, w, X, b):
        """
        This function calculates the hypothesis for the present model
        Argument:
         w -- weight vector
         X -- The input vector
         b -- The bias vector
        """

        H = self.sigmoid(np.dot(w.T, X) + b)
        return H

    def cost(self, H, Y, m):
        """
        This function calculates the cost of hypothesis
        Arguments:
         H -- The hypothesis vector
         Y -- The output
         m -- Number training samples
        """
        cost = -np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H)) / m
        cost = np.squeeze(cost)
        return cost

    def cal_gradient(self, w, H, X, Y):
        """
        Calculates gradient of the given model in learning space
        """
        m = X.shape[1]
        dw = np.dot(X, (H - Y).T) / m
        db = np.sum(H - Y) / m
        grads = {"dw": dw,
                 "db": db}
        return grads

    def gradient_position(self, w, b, X, Y):
        """
        It just gets calls various functions to get status of learning model
        Arguments:
         w -- weights, a numpy array of size (no. of features, 1)
         b -- bias, a scalar
         X -- data of size (no. of features, number of examples)
         Y -- true "label" vector (containing 0 or 1 ) of size (1, number of examples)
        """

        m = X.shape[1]
        H = self.hypothesis(w, X, b)  # compute activation
        cost = self.cost(H, Y, m)  # compute cost
        grads = self.cal_gradient(w, H, X, Y)  # compute gradient
        return grads, cost

    def gradient_descent(self, w, b, X, Y, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w - weights, a numpy array of size (num_px * num_px * 3, 1)
        b - bias, a scalar
        X -- data of size (no. of features, number of examples)
        Y -- true "label" vector (containing 0 or 1 ) of size (1, number of examples)
        print_cost - True to print the loss every 100 steps

        Returns:
        params - dictionary containing the weights w and bias b
        grads - dictionary containing the gradients of the weights and bias with respect to the cost function
        costs - list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """

        costs = []

        for i in range(self.num_iterations):
            # Cost and gradient calculation
            grads, cost = self.gradient_position(w, b, X, Y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule
            w -= (self.learning_rate * dw)
            b -= (self.learning_rate * db)

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print("Cost after iteration % i: %f" % (i, cost))

        params = {"w": w, "b": b}

        grads = {"dw": dw, "db": db}

        return params, grads, costs

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (n, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        X = np.array(X)
        m = X.shape[1]

        Y_prediction = np.zeros((1, m))

        w = self.w.reshape(X.shape[0], 1)
        b = self.b
        # Compute vector "H"
        H = self.hypothesis(w, X, b)

        for i in range(H.shape[1]):
            # Convert probabilities H[0,i] to actual predictions p[0,i]
            if H[0, i] >= 0.5:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0

        return Y_prediction

    def train_model(self, X_train, Y_train, X_test, Y_test, print_cost=False):
        """
        Builds the logistic regression model by calling the function you’ve implemented previously

        Arguments:
        X_train - training set represented by a numpy array of shape (features, m_train)
        Y_train - training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test - test set represented by a numpy array of shape (features, m_test)
        Y_test - test labels represented by a numpy array (vector) of shape (1, m_test)
        print_cost - Set to true to print the cost every 100 iterations

        Returns:
        d - dictionary containing information about the model.
        """
        # initialize parameters with zeros 
        dim = np.shape(X_train)[0]
        w, b = self.initialize_weight(dim)
        # Gradient descent 
        parameters, grads, costs = self.gradient_descent(w, b, X_train, Y_train, print_cost=False)

        # Retrieve parameters w and b from dictionary "parameters"
        self.w = parameters["w"]
        self.b = parameters["b"]

        # Predict test/train set examples 
        Y_prediction_test = self.predict(X_test)
        Y_prediction_train = self.predict(X_train)
        # Print train/test Errors
        train_score = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) *100
        test_score = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) *100
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) *100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) *100))
        d = {"costs": costs, "Y_prediction_test": Y_prediction_test, "Y_prediction_train": Y_prediction_train,
             "w": self.w, "b": self.b, "learning_rate": self.learning_rate, "num_iterations": self.num_iterations,
             "trainaccuracy": train_score, "test accuracy": test_score}

        return d

clf = MyLogisticRegression()
d = clf.train_model(X_train, Y_train, X_test, Y_test)
print(d["train accuracy"])
