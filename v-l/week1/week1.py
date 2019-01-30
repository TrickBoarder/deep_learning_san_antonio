# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:04:02 2019

@author: V.lous

tagging along with http://neuralnetworksanddeeplearning.com/chap1.html
"""
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 30, 10]) #input - hidden - output sizes  

net.SGD(training_data, 30, 10, 3.0, test_data=test_data) # 30 epochs, mini batch 10, 3.0 learning rate
# 9508/10000 = 95% classification rate  

net = network.Network([784, 100, 10]) #increase size of hidden layer  
net.SGD(training_data, 30, 10, 3.0, test_data=test_data) 
# book suggests that this improves classification rate. I got: 68% which is not good ...  

# exercise
'''
Try creating a network with just two layers - an input and an output layer, no hidden layer 
- with 784 and 10 neurons, respectively. Train the network using stochastic gradient descent. 
What classification accuracy can you achieve? 

epochs, minibatch, learn rate, accuracy
30, 10, 3.0: 75%
30, 10, 1.0: 76%
'''

net = network.Network([784, 10]) #no hidden layer
net.SGD(training_data, 30, 10, 1.0, test_data=test_data) 

