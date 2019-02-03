import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv as pseudo_inverse

def create_sine_data(min_x, max_x, n_datapoints = 300):
    X = np.linspace(min_x, max_x, n_datapoints)
    X = X.reshape(len(X), 1)
    y = np.sin(X)
    
    return X, y

# LOCALLY WEIGHTED REGRESSION: An example of non-parametric learning algorithm
class LWregressor:

    def __init__(self, max_iter = 1000, learning_rate = 0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        self.J = list()

    def compute_loss_and_gradient(self, iter_num, gradient = "batch"):
        grad = np.zeros_like(self.theta)
        
        if gradient == "batch":
            error = np.matmul(self.X, self.theta) - self.y
            cost = np.matmul(np.matmul(error.T, self.W), error).squeeze() / (2 * self.m)
            
            self.J.append([iter_num, float(cost)])
        else:
            pass
        return grad
    
    def normal_equation(self):
        _inverse = pseudo_inverse(np.matmul(np.matmul(self.X.T, self.W), X))
        _other = np.matmul(np.matmul(self.X.T, self.W), y)
        self.theta = np.matmul(_inverse, _other)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.theta = np.zeros((X.shape[1], 1))
        self.m = X.shape[0]
        
        self.mask = np.identity(self.m)
        self._all_considered = np.ones(self.m)
        self.W = np.multiply(self._all_considered, self.mask)

    def unweighted_plot(self):
        self._all_considered = np.ones(self.m)
        self.W = np.multiply(self._all_considered, self.mask)
        
        # Compute Optimal Theta
        self.normal_equation()
        
        # Compute target variable
        y_pred = np.matmul(self.X, self.theta)
        
        # Visualize the data
        plt.scatter(self.X[:, 1], self.y, s = 5, color = "red")
        plt.plot(self.X[:, 1], y_pred, color = "green")
        plt.title("Linear (Unweighted) Regression using Normal Equation")
        plt.xlabel("Input Feature")
        plt.ylabel("Predicted Value")
        plt.show()
    
    def predict(self, X):
        pass
    

if __name__ == "__main__":
    X, y = create_sine_data(-20, 20)
    X = np.hstack([np.ones((len(X), 1)), X])   # Intercept Column
    
    print(X.shape)
    print(y.shape)
    lr = LWregressor()
    lr.fit(X, y)
    lr.unweighted_plot()