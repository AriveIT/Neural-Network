#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

def initialize_parameters(dim):
    w = np.random.randn(dim, 1)*0.01
    b = 0
    return w, b

def sigmoid(z):
    return 1/(1+np.exp(-z))

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) 
    train_y = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:]) 
    test_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:]) 
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))
    
    return train_x, train_y, test_x, test_y, classes

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_parameters(dim):
    w = np.random.randn(dim, 1) *0.01
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    #calculate activation function
    A = sigmoid(np.dot(w.T, X)+b)

    #find the cost
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  

    #find gradient (back propagation)
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)
    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db} 
    return grads, cost

def gradient_descent(w, b, X, Y, iterations, learning_rate):
    costs = []
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        
        #update parameters
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        costs.append(cost)
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    print("After last iteration: %f" %(cost))

    params = {"w": w,
              "b": b}    
    return params, costs

def predict(w, b, X):    
    # number of example
    m = X.shape[1]
    y_pred = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X)+b)
    
    for i in range(A.shape[1]):
        y_pred[0,i] = 1 if A[0,i] >0.5 else 0
        pass
    return y_pred

def model(train_x, train_y, test_x, test_y, iterations, learning_rate):
    w, b = initialize_parameters(train_x.shape[0])
    parameters, costs = gradient_descent(w, b, train_x, train_y, iterations, learning_rate)
    
    w = parameters["w"]
    b = parameters["b"]
    
    # predict 
    train_pred_y = predict(w, b, train_x)
    test_pred_y = predict(w, b, test_x)
    print("Train Acc: {} %".format(100 - np.mean(np.abs(train_pred_y - train_y)) * 100))
    print("Test Acc: {} %".format(100 - np.mean(np.abs(test_pred_y - test_y)) * 100))
    
    return w, b, costs

def preprocessing(train_x, train_y, test_x, test_y):
    # Flatten out data
    train_x = train_x.reshape(train_x.shape[0], -1).T
    test_x = test_x.reshape(test_x.shape[0], -1).T

    # Center and standardize data
    train_x = train_x/255.
    test_x = test_x/255.

    return train_x, train_y, test_x, test_y

def init_model():
    # Load Dataset
    train_x, train_y, test_x, test_y, classes = load_dataset()

    train_x, train_y, test_x, test_y = preprocessing(train_x, train_y, test_x, test_y)

    w, b, costs = model(train_x, train_y, test_x, test_y, iterations = 500, learning_rate = 0.005)

    w = w.squeeze()

    strW = str(w)


def main():
    init_model()

#### Periphery Functions ####
def printCats():
    index = int(sys.argv[1])
    plt.imshow(train_x[index])
    plt.show()


############## Printing Cats ################
# index = int(sys.argv[1])
# plt.imshow(train_x[index])
# plt.show()

if __name__ == "__main__":
    main()