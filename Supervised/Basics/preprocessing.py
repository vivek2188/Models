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
