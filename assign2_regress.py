# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 07:59:33 2018

@author: Rebecca

Assignment 2 - Regressor
"""

"""
Import Library Section
"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sknn.mlp import Regressor, Layer
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, mean_absolute_error
import time
import numpy as np
import pandas as pd
"""
Load Data Section
"""
red_data = pd.read_csv("winequality-red.csv", encoding = 'utf-8')
x = red_data.iloc[:, 0:11]
y = red_data['quality']

"""
Pretreat Data Section
"""

np.random.seed(2016)

x_MinMax = preprocessing.MinMaxScaler()
y_MinMax = preprocessing.MinMaxScaler()

x.as_matrix(x)
x = np.array(x).reshape((len(x), 11))

y.as_matrix(y)
y = np.array(y).reshape((len(y), 1))

x = x_MinMax.fit_transform(x)
y = y_MinMax.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2016)

x_test.mean(axis=0)
y_test.mean(axis=0)
x_train.mean(axis=0)
y_train.mean(axis=0)

"""
Model - changing both activators and nodes
"""

start_time = time.time()
li = ["Rectifier", "Sigmoid", "Tanh", "ExpLin"]
print("ACTIVATOR1", "ACTIVATOR2", "NODE1", "NODE2", "TRAIN_MSE", "TEST_MSE")
for activator in li:
    for activator2 in li:
        node_li1 = [3,5,7,9]
        node_li2 = [8, 10, 12, 14]
        for node1 in node_li1:
            for node2 in node_li2:
                fit_trial = Regressor(
                        layers = [
                                Layer(activator, units = node1), 
                                Layer(activator2, units = node2),
                                Layer("Linear")], 
                                learning_rate = 0.02, 
                                random_state = 2016, 
                                n_iter = 100)
                fit_trial.fit(x_train, y_train)
                
                """
                Output Section
                """
                train_pred = fit_trial.predict(x_train)
                train_mse = mean_squared_error(train_pred, y_train)
                
            
                test_pred = fit_trial.predict(x_test)
                test_mse = mean_squared_error(test_pred, y_test)
                
                print(activator, activator2, node1, node2, train_mse, test_mse)               

stop_time = time.time()
print("Time Required for Optimization:", stop_time - start_time)

final_fit = Regressor(layers = [
        Layer("Tanh", units = 5),
        Layer("Rectifier", units = 10),
        Layer("Linear")],
        learning_rate = 0.02,
        random_state = 2016,
        n_iter = 500)
final_fit.fit(x_train, y_train)
final_pred = final_fit.predict(x_test)
final_mse = mean_squared_error(y_test, final_pred) #0.0167247
final_var = explained_variance_score(y_test, final_pred) #0.36920
final_error = mean_absolute_error(y_test, final_pred) #0.0967097
final_r2 = r2_score(y_test, final_pred) #0.3686637



