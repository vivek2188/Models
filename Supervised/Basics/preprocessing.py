import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
#print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# X,y are numpy ndarray(s)
print("Input Shape: {}, Output Shape: {}".format(X.shape, y.shape))
