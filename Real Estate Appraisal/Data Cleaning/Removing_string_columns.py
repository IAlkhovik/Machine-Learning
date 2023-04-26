#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import os

full_data = pd.read_csv("/Users/sergioruiz/Downloads/fullData.csv")
cleaned_data = pd.read_excel("/Users/sergioruiz/Downloads/data_cleaned/unnormalized/cleanedData.xlsx")

headers_full = list(full_data.columns)
headers_cleaned = list(cleaned_data.columns)

strings = full_data.select_dtypes(include="object").columns
cleaned_data = cleaned_data.drop(columns = strings)

print(cleaned_data.columns)
print(len(cleaned_data.columns))

writer = pd.ExcelWriter("/Users/sergioruiz/Downloads/cleanedDataWithStringFieldsGone.xlsx", engine = "xlsxwriter")
cleaned_data.to_excel(writer, sheet_name = "Sheet1", index = False)

writer.save()


# In[ ]:




