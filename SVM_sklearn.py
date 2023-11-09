import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load training and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Prepare data
X_train = train_data.drop(columns=["SNo", "Label"], axis=1)
X_test = test_data.drop("SNo", axis=1)

y_train = train_data["Label"]

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
classifier = SVC(kernel='linear', C=1000)
classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = classifier.predict(X_test)

# Create a DataFrame for predictions
df_predictions = pd.DataFrame({
    'SNo': range(1, len(y_pred) + 1),
    'Label': y_pred
})

# Save predictions to a CSV file
df_predictions.to_csv('sample_submission_1000.csv', index=False)
