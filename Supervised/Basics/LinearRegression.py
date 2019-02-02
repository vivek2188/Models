import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getData(filename):
    dataset = pd.read_csv(filename)
    print(dataset.head())
    '''
        The Output variable must be in the last row.
    '''
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values
    return X, y

if __name__ == "__main__":
    pass