# importing necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_test = train_test_split(X, y)

# training a linear SVM classifier
from sklearn.svm import SVC

svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

df_predictions = pd.DataFrame({
    'SNo': range(1, len(svm_predictions) + 1),  # Commence à 1 et continue jusqu'à la longueur de y_pred
    'Label': svm_predictions
})

df_predictions.to_csv('sample_submission.csv', index=False)

# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)