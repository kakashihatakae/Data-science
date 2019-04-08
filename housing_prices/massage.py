import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import joint_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
import category_encoders as ce
from math import sqrt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import matplotlib.pyplot as plt

imputer = SimpleImputer()
imputer_mode = SimpleImputer(strategy='most_frequent')

print("loading data")
joint- = pd.read_csv('~/python/kaggle/housing_prices/joint-.csv')
test = pd.read_csv('~/python/kaggle/housing_prices/test.csv')

#skewness 
(mu, sigma) = norm.fit(joint-['SalePrice'], )
sns.distplot(joint-['SalePrice'], fit=norm)
fig = plt.figure()
k = stats.probplot(joint['SalePrice'], plot=plt)
plt.show()

null_cols = joint-.columns[joint-.isnull().any()]
quantity_null_col = joint-.isnull().any().sum()
quantity_per_col  = joint-[null_cols].isnull().sum()

# print(quantity_per_col)
# print('')
# print("number of features with null columns")
# print(quantity_null_col)

#---------------
#---------------
#LotFrontage Imputation
imputer = SimpleImputer()
imputer_mode = SimpleImputer(strategy='most_frequent')

#imputer = SimpleImputer(Strategy = 'median')
imputed_data = imputer.fit_transform(train['LotFrontage'].values.reshape(2919, 1))
train['LotFrontage'] = imputed_data

#Alley
# train['Alley'] = train['Alley'].replace({'NA':1, 'Pave':2, 'Grvl':3})
train['Alley'].describe()
train['Alley'] = train['Alley'].fillna(1)
train['Alley'] = train['Alley'].replace({'Grvl':2, 'Pave':3})

#MasVnrType
# lb = LabelBinarizer()
imputed_data = imputer_mode.fit_transform(train['MasVnrType'].values.reshape(2919,1))
train['MasVnrType'] = imputed_data
#-------Street-----------
train['Street'] = train['Street'].replace({'Grvl':1, 'Pave':0})
#---------LotShape---------
train['LotShape'] = train['LotShape'].replace({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1})
#------LandContour
train['LandContour'] = train['LandContour'].replace({'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1})
#----------Utilities---------
train['Utilities'] = train['Utilities'].fillna('AllPub')
train['Utilities'] = train['Utilities'].replace({'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1})
#---------LandSlope
train['LandSlope'] = train['LandSlope'].replace({'Gtl':3, 'Mod':2, 'Sev':1})
#------MasVnrArea---
train['MasVnrArea'] = imputer.fit_transform(train['MasVnrArea'].values.reshape(2919,1))
#------------Exterqual-----------
train['ExterQual'] = train['ExterQual'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
#------ExternCond-------
train['ExterCond'] = train['ExterCond'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})
#-----BsmtQual------
train['BsmtQual'] = train['BsmtQual'].fillna(1)
train['BsmtQual'] = train['BsmtQual'].replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2})
#-----BsmtCond
train['BsmtCond'] = train['BsmtCond'].fillna(1)
train['BsmtCond'] = train['BsmtCond'].replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2})
#--------BsmtExposure
train['BsmtExposure'] = train['BsmtExposure'].fillna(1)
train['BsmtExposure'] = train['BsmtExposure'].replace({ 'Gd':5, 'Av':4, 'Mn':3, 'No':2})
#BsmtFinType1
train['BsmtFinType1'] = train['BsmtFinType1'].fillna(1)
train['BsmtFinType1'] = train['BsmtFinType1'].replace({'Unf':2, 'LwQ':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})
#BsmtFinType2
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(1)
train['BsmtFinType2'] = train['BsmtFinType2'].replace({'Unf':2, 'LwQ':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})
#Heating
train['HeatingQC'] = train['HeatingQC'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
#CentralAir
train['CentralAir'] = train['CentralAir'].replace({'Y':1, 'N':0})
#Electrical
train['Electrical'] = train['Electrical'].fillna(2)
train['Electrical'] = train['Electrical'].replace({'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5})
#kitchenQual
train['KitchenQual'] = train['KitchenQual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
#FireplceQu
train['FireplaceQu'] = train['FireplaceQu'].fillna(1)
train['FireplaceQu'] = train['FireplaceQu'].replace({'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
#GarageType
imputed_data_GYB = imputer_mode.fit_transform(train['GarageType'].values.reshape(2919,1))
train['GarageType'] = imputed_data_GYB
#GarageYrBlt
imputed_data_GYB = imputer_mode.fit_transform(train['GarageYrBlt'].values.reshape(2919,1))
train['GarageYrBlt'] = imputed_data_GYB
#GarageFinish
train['GarageFinish'] = train['GarageFinish'].fillna(1)
train['GarageFinish'] = train['GarageFinish'].replace({'Unf':2, 'RFn':3, 'Fin':4})
#GarageQual
train['GarageQual'] = train['GarageQual'].fillna(1)
train['GarageQual'] = train['GarageQual'].replace({'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
#GarageCond
train['GarageCond'] = train['GarageCond'].fillna(1)
train['GarageCond'] = train['GarageCond'].replace({'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
#PavedDrive
train['PavedDrive'] = train['PavedDrive'].replace({'N':1, 'P':2, 'Y':3})
#PoolQC
train['PoolQC'] = train['PoolQC'].fillna(1)
train['PoolQC'] = train['PoolQC'].replace({'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
#Fence
train['Fence'] = train['Fence'].fillna('None')
#BsmtFinSF1         1
imputed_data = imputer.fit_transform(train['BsmtFinSF1'].values.reshape(2919, 1))
train['BsmtFinSF1'] = imputed_data
# BsmtFinSF2         1
train['BsmtFinSF2'] = train['BsmtFinSF2'].fillna(0)
# BsmtFullBath       2
imputed_data = imputer_mode.fit_transform(train['BsmtFullBath'].values.reshape(2919,1))
train['BsmtFullBath'] = imputed_data
# BsmtHalfBath       2
imputed_data = imputer_mode.fit_transform(train['BsmtHalfBath'].values.reshape(2919,1))
train['BsmtHalfBath'] = imputed_data
# BsmtUnfSF          1
imputed_data = imputer.fit_transform(train['BsmtUnfSF'].values.reshape(2919, 1))
train['BsmtUnfSF'] = imputed_data
# GarageArea         1
imputed_data = imputer.fit_transform(train['GarageArea'].values.reshape(2919, 1))
train['GarageArea'] = imputed_data
# GarageCars         1
imputed_data = imputer_mode.fit_transform(train['GarageCars'].values.reshape(2919,1))
train['GarageCars'] = imputed_data
# KitchenQual        1
imputed_data = imputer_mode.fit_transform(train['KitchenQual'].values.reshape(2919,1))
train['KitchenQual'] = imputed_data
# TotalBsmtSF        1
imputed_data = imputer.fit_transform(train['TotalBsmtSF'].values.reshape(2919, 1))
train['TotalBsmtSF'] = imputed_data
# Utilities          2

#-----------------------------------------------------------------


lis_one_hot = ['MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                'BldgType', 'HouseStyle', 'RoofMatl','RoofStyle', 'Exterior1st', 'Exterior2nd',
                'MasVnrType', 'Foundation', 'Heating', 'Functional', 'GarageType', 'Fence',
                'MiscFeature', 'SaleType', 'SaleCondition']

one_hot = pd.get_dummies(train-, columns=lis_one_hot)
# print(one_hot.describe())

# path = '~/python/kaggle/one_hot.csv'
# one_hot.to_csv(path)

Y = one_hot['SalePrice']
one_hot.drop(['Id', 'SalePrice'], axis=1)

num_trees = 100
max_fet = 3
rr = RandomForestRegressor(n_estimators=num_trees, max_features=max_fet)

rr.fit(one_hot, Y)
predictions = rr.predict(one_hot)

print(sqrt(mean_squared_error(Y, predictions)))