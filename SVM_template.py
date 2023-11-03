import numpy as np

class MySVC:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape

    def initialize_weights(self, n_features):
        """
        Initialize weights and bias
        """
        self.w = np.zeros(n_features)
        self.b = 0

    def hinge_loss(self, X, y):
        """
        Compute the hinge loss
        """
        loss = 1 - y * (np.dot(X, self.w) + self.b)
        loss[loss < 0] = 0  # Set negative loss to zero
        return loss

    def gradient(self, X, y):
        """
        Compute the gradient of the hinge loss
        """
        margin = y * (np.dot(X, self.w) + self.b)
        dw = np.zeros(self.w.shape)
        db = 0

        if np.max(margin) < 1:
            dw = -np.dot(X.T, y * margin)
            db = -np.sum(y * margin)

        return dw, db

    def train(self, X, y):
        """
        Train the SVM classifier
        """
        self.initialize_weights(X.shape[1])

        for _ in range(self.num_iterations):
            dw, db = self.gradient(X, y)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        """
        Predict using the trained model
        """
        return np.sign(np.dot(X, self.w) + self.b)

    def accuracy(self, X, y):
        """
        Calculate accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Assuming X_train, y_train, X_test, y_test are your training and testing data
# If not, replace them with your actual data

# Step 1: Preprocess the data if needed (e.g., scaling, adding bias term)

# Step 2: Initialize and train the model
svc = MySVC()
svc.train(X_train, y_train)

# Step 3: Predict on test set and calculate accuracy
test_accuracy = svc.accuracy(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")