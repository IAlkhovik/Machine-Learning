import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./data_cleaned/cleanedAndTrimmedData.xlsx')

price = df['SalePrice']
year = df['YrSold']
plt.scatter(year,price)
plt.show()