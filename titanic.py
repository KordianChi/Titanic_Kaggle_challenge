# To tylko projekt testowy
# Prosty projekt oparty o dane kaggle

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'train.csv')

y = df['Survived']
df.drop('Survived', axis='columns', inplace=True)

df.rename(columns={'Pclass': 'pclass', 'Sex': 'sex', 'Age': 'age', 'SibSp': 'sibsp', 'Parch': 'parch', 'Fare': 'fare',
                   'Embarked': 'embarked'}, inplace=True)

df.drop(['PassengerId', 'Ticket'], axis='columns', inplace=True)

df['title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df.drop(['Name'], axis='columns', inplace=True)

df['has_cabin'] = df['Cabin'].str.get(0)
df.drop(['Cabin'], axis='columns', inplace=True)

median_age = df['age'].median()
mode_embarked = df['embarked'].mode()

df['age'].fillna(median_age, inplace=True)
df['has_cabin'].fillna('uknown', inplace=True)
df['embarked'].fillna(mode_embarked, inplace=True)
df['fare'].fillna(0, inplace=True)

IQR = (df['fare'].quantile(q=0.75) - df['fare'].quantile(q=0.25))
cat_4 = min(max(df['fare']), df['fare'].median() + 1.5*IQR)
cat_2 = df['fare'].median()
cat_3 = df['fare'].quantile(q=0.75)
cat_1 = df['fare'].quantile(q=0.25)

df['fare'].where(df['fare'] >= cat_1, 1, inplace=True)

df['fare'].where((df['fare'] == 1) | (df['fare'] >= cat_2), 2, inplace=True)

df['fare'].where((df['fare'] == 1) | (df['fare'] == 2) | (df['fare'] >= cat_3), 3, inplace=True)
df['fare'].where((df['fare'] == 1) | (df['fare'] == 2) | (df['fare'] == 3) | (df['fare'] >= cat_4), 4, inplace=True)

df['fare'].where((df['fare'] == 1) | (df['fare'] == 2) | (df['fare'] == 3) | (df['fare'] == 4), 5, inplace=True)
df['fare'] = df['fare'].astype(int)

IQR = (df['age'].quantile(q=0.75) - df['age'].quantile(q=0.25))
cat_4 = min(max(df['age']), df['age'].median() + 1.5*IQR)
cat_2 = df['age'].median()
cat_3 = df['age'].quantile(q=0.75)
cat_1 = df['age'].quantile(q=0.25)
cat_0 = max(min(df['age']), df['age'].median() - 1.5*IQR)

df['age'].where(df['age'] >= cat_0, 1, inplace=True)

df['age'].where((df['age'] == 1) | (df['age'] >= cat_1), 2, inplace=True)

df['age'].where((df['age'] == 1) | (df['age'] == 2) | (df['age'] >= cat_2), 3, inplace=True)
df['age'].where((df['age'] == 1) | (df['age'] == 2) | (df['age'] == 3) | (df['age'] >= cat_3), 4, inplace=True)

df['age'].where((df['age'] == 1) | (df['age'] == 2) | (df['age'] == 3) | (df['age'] == 4) | (df['age'] >= cat_4), 5,
                inplace=True)
df['age'].where((df['age'] == 1) | (df['age'] == 2) | (df['age'] == 3) | (df['age'] == 4) | (df['age'] == 5), 6,
                inplace=True)
df['age'] = df['age'].astype(int)

df['title'].where((df['title'] == 'Mr') | (df['title'] == 'Miss') | (df['title'] == 'Mrs') | (df['title'] == 'Master'),
                  'Other', inplace=True)

one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(df)
X = one_hot_encoder.transform(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=112)

tree_params = {'n_estimators': [50, 100, 200, 300], 'criterion': ['gini', 'entropy'],
               'max_features': ['auto', 'sqrt', 'log2']}

model_rfc = GridSearchCV(RandomForestClassifier(), tree_params, cv=5, scoring='accuracy', verbose=3)
model_rfc.fit(X_train, y_train)
y_rfc = model_rfc.predict(X_test)
print(classification_report(y_test, y_rfc))
print(model_rfc.best_params_)
result_rfc = pd.Series(y_rfc)
filepath = Path(r'C:\Users\PC\Desktop\Titanic\result_rfc.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
result_rfc.to_csv(filepath)
