import pandas as pd
import numpy as np

# Load train and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocess the data
train_df = train_df.drop(['Id'], axis=1)
test_df = test_df.drop(['Id'], axis=1)


# Prepare the data for training
def svm_train(X, Y, epochs, alpha):
    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(1, epochs + 1):
        for i, x in enumerate(X):
            condition = Y[i] * (np.dot(x, w) - b) >= 1
            if condition:
                w = w - alpha * (2 * 1 / epoch * w)
            else:
                w = w + alpha * (np.dot(x, Y[i]) - 2 * 1 / epoch * w)
                b = b + alpha * Y[i]

    return w, b

def svm_predict(X, w, b):
    return np.sign(np.dot(X, w) - b)

classifiers = {}

for class_name in s:
    # Convert labels to binary for both train and test sets
    Y_train = np.where(train_df['Species'] == class_name, -1, 1)
    Y_test = np.where(test_df['Species'] == class_name, -1, 1)

    X_train = train_df.drop(['Species'], axis=1).values
    X_test = test_df.drop(['Species'], axis=1).values

    # Train SVM classifier
    w, b = svm_train(X_train, Y_train, epochs=10000, alpha=0.0001)

    classifiers[class_name] = {'w': w, 'b': b}

# Evaluate on test set
for class_name, info in classifiers.items():
    Y_true = np.where(test_df['Species'] == class_name, -1, 1)
    Y_pred = svm_predict(X_test, info['w'], info['b'])
    accuracy = np.mean(Y_true == Y_pred)
    print(f'Class: {class_name}, Accuracy: {accuracy}')