import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

titanic_data = pd.read_csv('~/python/kaggle/train.csv')
# print(titanic_data.describe() 
Y = titanic_data.Survived


titanic_data['Sex'] = titanic_data['Sex'].replace({"female":0,"male": 1})
imputer = SimpleImputer()
imputed_data = imputer.fit_transform(titanic_data['Age'].reshape(891,1))
# print(imputed_data.shape)
# print(titanic_data['Age'].shape)
titanic_data['Age'] = imputed_data.reshape(891,)

features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age']
X = titanic_data[features]

model = LogisticRegression()
tree = DecisionTreeClassifier()

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state = 0)
model.fit(train_X,train_Y)
tree.fit(train_X, train_Y)

Y_true = val_Y
Y_pred = model.predict(val_X)
tree_pred = tree.predict(val_X)

print("the score is")
print(accuracy_score(Y_true, Y_pred))

print("tree")
print(accuracy_score(Y_true, tree_pred))