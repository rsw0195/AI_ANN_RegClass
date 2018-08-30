# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 07:58:25 2018

@author: Rebecca

Homework 2 - Classification
"""

"""
import packages
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import time


white_data = pd.read_csv("winequality-white.csv", encoding = 'utf-8', delimiter = ';')
x = white_data.iloc[:, 0:11]
qual = white_data['quality']

white_data['labs'] = pd.cut(qual, bins = ([2, 4, 7, 9]), labels = ['low', 'medium', 'high'])
y = white_data['labs']

x, y = x.as_matrix(), y.as_matrix()

x_train, x_test, y_train, y_test = x[:3500], x[3500:], y[:3500], y[3500:]

shuffle_index = np.random.permutation(3500)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


start_time = time.time()
size = [10, 20, 30, 40, 50]
activators = ['relu', 'tanh', 'logistic', 'identity']
solvers = ['lbfgs', 'sgd', 'adam']
print("ACTIVATOR", "SOLVER", "LAYER_SIZE", "TRAIN_SCORE", "TEST_SCORE")
for act in activators:
    for siz in size:
        for solv in solvers:
            mlp = MLPClassifier(activation = act, hidden_layer_sizes = (siz,), max_iter = 50, alpha = 1e-4, solver = solv, 
                    verbose = False, tol = 1e-4, random_state = 2016)
            pred = mlp.fit(x_train, y_train).predict(x_test)
            
            train_score = mlp.score(x_train, y_train)
            test_score = mlp.score(x_test, y_test)
            
            print(act, solv, siz, train_score, test_score)
stop_time = time.time()                      

final_mlp = MLPClassifier(activation = 'tanh', hidden_layer_sizes = (50,), max_iter = 50, alpha = 1e-4, solver = 'lbfgs', 
                    verbose = False, tol = 1e-4, random_state = 2016)
final_fit = final_mlp.fit(x_train, y_train)
final_pred = final_fit.predict(x_test)

print("Train Set Score: %f" % final_mlp.score(x_train, y_train)) 
print("Test Set Score: %f" % final_mlp.score(x_test, y_test))