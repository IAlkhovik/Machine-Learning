import pandas as pd
import numpy as np

for x in range(3):
    #read data
    df = pd.read_csv('./data/train.csv')

    #replace all NA and nan with 0 (a better way could be to use a linear regression to predict missing data)
    df = df.replace('NA', 0)
    df = df.replace(np.nan, 0)

    salePrice_unnormalized = df['SalePrice']

    #define all sets of columns
    allCols = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
        'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
        'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
        'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
        'SaleCondition', 'SalePrice']
    numericalCols = ['Id', 'MSSubClass','LotFrontage','LotArea', 'OverallQual',
                    'OverallCond', 'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF2',
                    'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                    'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                    'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars',
                    'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch', 
                    'ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice']
    colsToRemove = ['Id','Street','Alley','LandContour','Utilities','LandSlope','Neighborhood',
                    'Condition1','Condition2','RoofMatl','ExterQual','ExterCond','BsmtQual',
                    'BsmtCond','Heating','CentralAir','LowQualFinSF','KitchenQual',
                    'Functional','FireplaceQu','GarageQual','GarageCond','PavedDrive',
                    'PoolQC','SaleType','SaleCondition']
    veryTrimmedCols = ['LotFrontage','LotArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageArea','SalePrice','salePrice_unnormalized']

    #find all non numerical columns
    colsToClean = allCols
    for i in range(len(numericalCols)):
        colsToClean.remove(numericalCols[i])

    #assign integer numbers to each unique string in each text column
    for i in range(len(colsToClean)):
        thisCol = colsToClean[i]
        uniqueValues = df[thisCol].unique()
        for j in range(len(uniqueValues)):
            df[thisCol] = df[thisCol].replace(uniqueValues[j], j)

    #normalize
    if(x==0):
        df=(df-df.min())/(df.max()-df.min()) #number between 0 and 1
        folder = "normalized_zeroToOne"
    elif(x==2):
        df=(df-df.mean())/df.std() #mean 0, standard deviation of 1
        folder = "normalized_normal"
    else:
        folder = "unnormalized"

    print(df['SalePrice'].mean())
    print(df['SalePrice'].std())

    df['salePrice_unnormalized'] = salePrice_unnormalized
    
    #trim all extraneous data (not enough uniqueness to be worth analyzing)
    df_trimmed = df
    for i in range(len(colsToRemove)):
        df_trimmed = df_trimmed.drop(colsToRemove[i],axis=1)

    #just include a few columns that are important
    df_veryTrimmed = df[veryTrimmedCols]

    #send each dataframe to excel
    name1 = "data_cleaned/" + folder + "/cleanedData.xlsx"
    df.to_excel(name1)
    df_trimmed.to_excel("data_cleaned/" + folder + "/cleanedAndTrimmedData.xlsx")
    df_veryTrimmed.to_excel("data_cleaned/" + folder + "/cleanedAndVeryTrimmedData.xlsx")