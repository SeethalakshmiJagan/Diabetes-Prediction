import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

# Load the dataset without specifying column names
pima_df = pd.read_csv("diabetes.csv")

# Specify the column names
pima_df.columns=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'target']

# Define features and target variable
X = pima_df.drop('target', axis=1)
Y = pima_df['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=101)

# Fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_predict = model.predict(X_test)

# Evaluate the model
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))
print(metrics.classification_report(y_test, y_predict))

# Save the model to a file
pickle.dump(model, open('model.pkl', 'wb'))
