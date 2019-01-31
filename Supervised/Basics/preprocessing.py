import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
#print(dataset.head())
print(dataset.isna().any())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# X,y are numpy ndarray(s)
print("Input Shape: {}, Output Shape: {}".format(X.shape, y.shape))

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbl_x = LabelEncoder()
X[:, 0] = lbl_x.fit_transform(X[:, 0])
ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()
lbl_y = LabelEncoder()
y = lbl_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
print("Training Shape: {}, Test Shape: {}".format(X_train.shape, X_test.shape))

from sklearn.preprocessing import StandardScaler
scl_x = StandardScaler()
X_train[:, 3:] = scl_x.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scl_x.transform(X_test[:, 3:])
print(X_train)
