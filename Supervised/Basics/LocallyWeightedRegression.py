import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv as pseudo_inverse

def create_sine_data(min_x, max_x, n_datapoints = 500):
    X = np.linspace(min_x, max_x, n_datapoints)
    X = X.reshape(len(X), 1)
    y = np.sin(X)
    
    return X, y

# LOCALLY WEIGHTED REGRESSION: An example of non-parametric learning algorithm
class LWregressor:

    def __init__(self, bandwidth_parameter = 5):
        self.bandwidth_parameter = bandwidth_parameter
        
        self.J = list()
        
    def normal_equation(self):
        _inverse = pseudo_inverse(np.matmul(np.matmul(self.X.T, self.W), self.X))
        _other = np.matmul(np.matmul(self.X.T, self.W), y)
        self.theta = np.matmul(_inverse, _other)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.theta = np.zeros((self.X.shape[1], 1))
        self.m = X.shape[0]
        
        self.mask = np.identity(self.m)
        self._all_considered = np.ones(self.m)
        self.W = np.multiply(self._all_considered, self.mask)

    def predict(self, x = None, type_reg = "unweighted"):
        if type_reg == "unweighted":
            self._all_considered = np.ones(self.m)
            self.W = np.multiply(self._all_considered, self.mask)

            self.normal_equation() 
            self.y_pred = np.matmul(self.X, self.theta)
        
            plt.plot(self.X[:, 1], self.y_pred, color = "green")
            plt.title("Linear (Unweighted) Regression using Normal Equation")
            
        elif type_reg == "weighted":
            if x is None:
                print("ERROR: Input not provided")
                return
            
            t = np.array(x) - self.X
            self._all_considered =  - np.matmul(t, t.T) / (2 * self.bandwidth_parameter * self.bandwidth_parameter)
            self._all_considered = np.exp(self._all_considered)
            self.W = np.multiply(self._all_considered, self.mask)
            
            self.normal_equation()
            self.y_pred = np.matmul(X, self.theta)
            
            plt.plot(X[:, 1], self.y_pred, color = "blue")
            plt.title("Locally Weighted Regression (bandwidth parameter = {})".format(self.bandwidth_parameter))
        else:
            print("ERROR: Enter valid \"plot\" parameter")
            return

    def visualize(self):
        plt.xlabel("Input Feature")
        plt.ylabel("Predicted Value")
        plt.scatter(self.X[:, 1], self.y, s = 5, color = "red")
        plt.show()

        
if __name__ == "__main__":
    X, y = create_sine_data(-20, 20)
    X = np.hstack([np.ones((len(X), 1)), X])   # Intercept Column
    
    print("Input Shape: {}".format(X.shape))
    print("Ouput Shape: {}".format(y.shape))
    lr = LWregressor(bandwidth_parameter = 0.502)
    lr.fit(X, y)
    lr.predict(x = X[0], type_reg = "weighted")
    lr.visualize()