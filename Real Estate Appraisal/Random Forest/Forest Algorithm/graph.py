import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("output.csv", sep=",")
data.loc[:, ['Actual', 'Prediction']]

plt.scatter(data['Actual'], data['Prediction'])
b, a = np.polyfit(data['Actual'], data['Prediction'], 1)
plt.plot(data['Actual'], b*data['Actual'] + a)
plt.plot([0,10], [0,10])
plt.title("Random Forest Predictions - Trimmed Data")
plt.xlabel("Correct Value")
plt.ylabel("Predicted Value")
plt.legend(['Predictions', 'Regression', 'Correct'])
plt.show()