import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
#from sklearn.tree import export_graphviz
#from IPython.display import Image
#import graphviz
from sklearn.model_selection import RandomizedSearchCV, train_test_split

#print("DATA READ")
data = pd.read_csv("PCA_95_Variance.csv", sep=",")
cols = [str(i) for i in range(61)]
cols.append('SalePricev2')
data = data.loc[:, cols]

#print("TARGET IDENTIFICATION")
X = data.drop('SalePricev2', axis=1)
y = data['SalePricev2']

#print("DATA SPLIT")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print("FOREST")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#print("PREDICT")
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#print("OUTPUT")
df = pd.DataFrame()
df['Actual'] = y_test
df['Prediction'] = y_pred
df.to_excel("output.xlsx")