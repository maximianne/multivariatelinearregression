import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv

# Function for the mulitvariate linear regression
# Y= B_0*x_0+ B_1*x_1+ ... + B_n*x_n + e 

# for BGD
def cost_function_MSE(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y)**2)/(2*m)
    return J

def batch_gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
 
    for iteration in range(iterations):
    # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function_MSE(X, Y, B)
        cost_history[iteration] = cost
 
    return B, cost_history

# for Matrix 
def solve_betas(x,y): 
    n,m = x.shape
    X0 = np.ones((n,1))
    Xnew = np.hstack((X0,x))
    x_t = Xnew.T
    x_t_x = np.matmul(x_t, Xnew)
    x_t_x_i =  inv(x_t_x)
    x_t_y = np.matmul(x_t, y)
    return np.matmul(x_t_x_i, x_t_y)

# Use for both 
def y_predict(B, x): 
    y_pred = []
    for i in range(0, len(X)): 
        cur = X[i]
        temp = Beta[0]
        for j in range(0, len(cur)):
            temp += cur[j]*Beta[j+1]
        y_pred.append(temp)
    return y_pred

def r_sqaure(y, y_pred):
    y = y.tolist()
    y_mean = np.sum(y)/len(y)
    sum = 0
    for i in range(0, len(y)): 
        y_val = y[i]
        y_calc = y_pred[i]
        diff = (y_val- y_calc)**2
        sum +=diff

    sum2 = 0
    for i in range(0, len(y)): 
        y_val = y[i]
        diff = (y_val- y_mean)**2
        sum2 +=diff

    return 1-(sum/sum2)

# Test cases
data = pd.read_excel('energy.xlsx')

X = data.iloc[:,:4]
y = data.iloc[:,-1]

Beta = solve_betas(X,y)
print(Beta)

#using gradient descent 
sc = StandardScaler()
X = sc.fit_transform(X)

m = 7000
f = 4
X_train = X[:m,:f]
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
y_train = y[:m]
X_test = X[m:,:f]
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
y_test = y[m:]

# Initial Coefficients
B = np.zeros(X_train.shape[1])
alpha = 0.005
iter_ = 2000
newB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter_)

print(newB)

y_pred_equation = y_predict(Beta, y)
y_pred_bgd= y_predict(newB, y)

print(r_sqaure(y, y_pred_equation))
print(r_sqaure(y,y_pred_bgd))