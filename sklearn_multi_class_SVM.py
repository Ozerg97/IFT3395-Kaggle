from sklearn import datasets
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd

#Here in the above, we can see that we have just called SVC models, the OneVsRestClassifier model, and iris data. Let’s make independent and target variables using the iris data.

iris = pd.read_csv('train.csv')
X = iris.data
y = iris.target

#Let’s instantiate the SVC model.

svc = SVC()

#Instantiating the One-Vs-Rest Classifier for the SVC model.

o_vs_r = OneVsRestClassifier(svc)

#Let’s train the model

o_vs_r.fit(X, y)

#Predicting values from the model.

yhat = o_vs_r.predict(X)
print(yhat)
#Output:


#Here in the above, we can see that the support vector classifier is predicting 3 classes and we did not require to use three binary classifiers.  Let’s take a closer look at One-vs-One (OvO).