import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('train.csv')
df = df.drop(['Id'], axis=1)

# Remove rows 100 to 149
rows = list(range(100, 150))
df = df.drop(df.index[rows])

# Extract features and target values
X = df[['SepalLengthCm', 'PetalLengthCm']].values
target = df['Species']
Y = np.where(target == 'Iris-setosa', -1, 1)

# Shuffle the data
np.random.seed(0)  # Set a seed for reproducibility
shuffle_idx = np.random.permutation(len(Y))
X, Y = X[shuffle_idx], Y[shuffle_idx]

# Split the data into training and test sets
split_ratio = 0.9
split_idx = int(len(X) * split_ratio)
x_train, x_test = X[:split_idx], X[split_idx:]
y_train, y_test = Y[:split_idx], Y[split_idx:]

# Initialize SVM parameters
w = np.zeros(X.shape[1])
b = 0
learning_rate = 0.0001
epochs = 10000

# Training the SVM
for epoch in range(1, epochs + 1):
    for i, x in enumerate(x_train):
        condition = y_train[i] * (np.dot(x, w) - b)
        if condition >= 1:
            w -= learning_rate * (2 * 1 / epoch * w)
        else:
            w -= learning_rate * (x * y_train[i] - 2 * 1 / epoch * w)
            b -= learning_rate * -y_train[i]

# Make predictions on the test set
predictions = np.sign(np.dot(x_test, w) - b)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print("Custom SVM Accuracy:", accuracy)

# 3-class classification using One-vs-Rest (OvR) strategy

# Load the data again
df = pd.read_csv('train.csv')
df = df.drop(['Id'], axis=1)

# Initialize a list to store the accuracy of each class
accuracies = []

for target_class in df['Species'].unique():
    df_class = df.copy()
    df_class['Species'] = np.where(df_class['Species'] == target_class, 1, -1)
    X = df_class[['SepalLengthCm', 'PetalLengthCm']].values
    Y = df_class['Species'].values

    # Shuffle the data
    np.random.seed(0)  # Set a seed for reproducibility
    shuffle_idx = np.random.permutation(len(Y))
    X, Y = X[shuffle_idx], Y[shuffle_idx]

    # Split the data
    split_idx = int(len(X) * split_ratio)
    x_train, x_test = X[:split_idx], X[split_idx:]
    y_train, y_test = Y[:split_idx], Y[split_idx:]

    # Initialize SVM parameters
    w = np.zeros(X.shape[1])
    b = 0
    learning_rate = 0.0001
    epochs = 10000

    # Training the SVM
    for epoch in range(1, epochs + 1):
        for i, x in enumerate(x_train):
            condition = y_train[i] * (np.dot(x, w) - b)
            if condition >= 1:
                w -= learning_rate * (2 * 1 / epoch * w)
            else:
                w -= learning_rate * (x * y_train[i] - 2 * 1 / epoch * w)
                b -= learning_rate * -y_train[i]

    # Make predictions on the test set
    predictions = np.sign(np.dot(x_test, w) - b)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    accuracies.append(accuracy)

print("3-Class Classifier Accuracies:", accuracies)