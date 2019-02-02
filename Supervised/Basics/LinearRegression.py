import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getData(filename):
    dataset = pd.read_csv(filename)
    print(dataset.isna().any())
    print(dataset.head())
    '''
        The Output variable must be in the last row.
    '''
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values
    return X, y

class LinearRegression:
    def __init__(self, max_iter = 1000, learning_rate = 0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        self.J = list()
    
    def compute_cost_and_gradient(self):
        hypothesis = np.matmul(self.X, self.theta)
        _ = np.sum(np.square(hypothesis - self.y)) / (2 * self.m)
        grad = np.matmul(self.X.T, hypothesis - self.y) / self.m
        return _, grad
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.theta = np.zeros((X.shape[1], 1))
        self.m = X.shape[0]
        
        # Applying Batch Gradient Descent
        for itr in range(self.max_iter):
            itr_cost, gradient = self.compute_cost_and_gradient()
            self.theta = self.theta - self.learning_rate * gradient
            self.J.append(itr_cost)
        
    def predict(self, X):
        return np.matmul(X, self.theta)
    

if __name__ == "__main__":
    X, y = getData("Salary_Data.csv")
    m = X.shape[0] # Number of training examples

    #Adding intercept column
    X = np.hstack([np.ones((m,1)), X])
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)