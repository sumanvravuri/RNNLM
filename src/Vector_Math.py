'''
Created on Aug 24, 2014

@author: sumanravuri
'''
from scipy.special import expit
import numpy as np

class Vector_Math:
    #math functions
    def sigmoid(self,inputs): #completed, expensive, should be compiled
        return expit(inputs)
        #return 1./(1.+np.exp(-inputs)) #1/(1+e^-X)
    def softmax(self, inputs): #completed, expensive, should be compiled
        exp_inputs = np.exp(inputs - np.max(inputs,axis=1)[:,np.newaxis])
        return exp_inputs / np.sum(exp_inputs, axis=1)[:, np.newaxis]
    def weight_matrix_multiply(self,inputs,weights,biases): #completed, expensive, should be compiled
        out = np.dot(inputs,weights)
        out += biases
        return out