import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer

imputer = SimpleImputer()
imputer_mode = SimpleImputer(strategy='most_frequent')

print("loading data")
train = pd.read_csv('~/python/kaggle/housing_prices/train.csv')
test = pd.read_csv('~/python/kaggle/housing_prices/test.csv')

print(train.describe())

null_cols = train.columns[train.isnull().any()]
quantity_null_col = train.isnull().any().sum()
quantity_per_col  = train[null_cols].isnull().sum()

print(quantity_per_col)
print('')
print("number of features with null columns")
print(quantity_null_col)

#---------------
#LotFrontage Imputation


#imputer = SimpleImputer(Strategy = 'median')
train['LotFrontage'] = imputer.fit_transform(train['LotFrontage'].reshape(1460, 1))
print(train['LotFrontage'].shape)

#Alley
train['Alley'].describe()
train['Alley'] = train['Alley'].fillna(1)
train['Alley'] = train['Alley'].replace({'Grvl':2, 'Pave':3})
# print(train['Alley'])

#MasVnrType

lb = LabelBinarizer()
imputed_data = imputer_mode.fit_transform(train['MasVnrType'].values.reshape(1460,1))
train['MasVnrType'] = imputed_data

one_hot_encoded = lb.fit_transform(train['MasVnrType'])
df = pd.DataFrame(one_hot_encoded, columns=lb.classes_)
concatenated = pd.concat([train, df], axis=1)
train = concatenated
#code to drop the masvnrtype columns
print(train.head())

#-------Street-----------
train['Street'] = train['Street'].replace({'Grvl':1, 'Pave':0})

#---------LotShape---------
train['LotShape'] = train['LotShape'].replace({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1})

#------LandContour
train['LandContour'] = train['LandContour'].replace({'Lvl':4, 'Bnk':3, 'Hls':2, 'Low':1})

#----------Utilities---------
train['Utilities'] = train['Utilities'].replace({'AllPub':4, 'NoSewr':3, 'NoSewa':2, 'ELO':1})

#---------LandSlope
train['LandSlope'] = train['LandSlope'].replace({'Gtl':3, 'Mod':2, 'Sev':1})
print(train['LandSlope'])

#------MasVnrArea---
train['MasVnrArea'] = imputer.fit_transform(train['MasVnrArea'].values.reshape(1460,1))

#------------Exterqual-----------
train['ExterQual'] = train['ExterQual'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})

#------ExternCond-------
train['ExterCond'] = train['ExterCond'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})

#-----BsmtQual------
train['BsmtQual'] = train['BsmtQual'].fillna(1)
train['BsmtQual'] = train['BsmtQual'].replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2})

#-----BsmtCond
train['BsmtCond'] = train['BsmtCond'].fillna(1)
train['ExterCond'] = train['BsmtCond'].replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2})

#--------BsmtExposure
train['BsmtEposure'] = train['BsmtExposure'].fillna(1)
train['BsmtExposure'] = train['BsmtExposure'].replace({ 'Gd':5, 'Av':4, 'Mn':3, 'No':2})

#BsmtFinType1
train['BsmtFinType1'] = train['BsmtFinType1'].fillna(1)
train['BsmtFinType1'] = train['BsmtFinType1'].replace({'Unf':2, 'Lwq':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})

#BsmtFinType2
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(1)
train['BsmtFinType2'] = train['BsmtFinType2'].replace({'Unf':2, 'Lwq':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})

#Heating
train['HeatingQC'] = train['HeatingQC'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})

#CentralAir
train['CentralAir'] = train['CentralAir'].replace({'Y':1, 'N':0})

#Electrical
train['Electrical'] = train['Electrical'].replace({'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5})

#kitchenQual
train['KitchenQual'] = train['KitchenQual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})

#FireplceQu
train['FireplaceQu'] = train['FireplaceQu'].fillna(1)
train['FireplaceQu'] = train['FireplaceQu'].replace({'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})

#GarageYrBlt
imputed_data_GYB = imputer_mode.fit_transform(train['GarageType'].values.reshape(1460,1))
train['GarageType'] = imputed_data_GYB

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

