import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

titanic_data = pd.read_csv('~/python/kaggle/train.csv')
# print(titanic_data.describe())

Y = titanic_data.Survived


titanic_data['Sex'] = titanic_data['Sex'].replace({"female":0,"male": 1})
imputer = SimpleImputer()

age = titanic_data['Age']
age = age.values.reshape(891,1)

imputed_data = imputer.fit_transform(age)
titanic_data['Age'] = imputed_data.reshape(891,)

features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Sex', 'Age']
X = titanic_data[features]

# model = LogisticRegression()

seed = 1
num_trees = 100
max_fet = 3
tree = RandomForestClassifier(n_estimators=num_trees, max_features=max_fet)

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state = 0)
# model.fit(train_X,train_Y)
tree.fit(train_X, train_Y)

Y_true = val_Y
# Y_pred = model.predict(val_X)
tree_pred = tree.predict(val_X)

# print("the score is")
# print(accuracy_score(Y_true, Y_pred))

print("Random forest")
print(accuracy_score(Y_true, tree_pred))

print("working on the test data")

test = pd.read_csv('~/python/kaggle/test.csv')
test['Sex'] = test['Sex'].replace({"female":0,"male": 1})

imputer = SimpleImputer()

age_test = test['Age']
age_test = age_test.values.reshape(418,1)

imputed_data = imputer.fit_transform(age_test)
test['Age'] = imputed_data.reshape(418,)

predictions = tree.predict( test[features] )

print(predictions)

submission = test['PassengerId']
# submission = submission.append(pd.DataFrame(predictions))
submission = pd.concat([submission, pd.DataFrame(predictions)], axis=1)
print(submission)
submission.to_csv('submission.csv', sep=',', encoding='utf-8',index=False)