from sklearn.linear_model import LogisticRegression
import numpy as np


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_gradient_db(X, y, w, b):
# computes derivative of log-loss function with respect to b
# using batch gradient descent
  w = np.squeeze(w)
  m = X.shape[0]
  db = 0

  for i in range(m):
       
     pred = sigmoid(np.dot(X[i], w) + b)  
     error = pred - y[i]
     db = db + error

  db = db / m

  return db

class LR_u(LogisticRegression):
   
   def update_intercept(self, X, y):
       
       w, b = self.coef_, self.intercept_
       lr = 0.0001
       num_iters = 1

       for i in range(num_iters):
          db = compute_gradient_db(X, y, w, b)
          b = b - lr*db

       self.intercept_ = b
 
       return self