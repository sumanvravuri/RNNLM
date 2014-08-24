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
        #subtracting max value of each data point below for numerical stability
        #exp_inputs = np.exp(inputs - np.transpose(np.tile(np.max(inputs,axis=1), (inputs.shape[1],1))))
        exp_inputs = np.exp(inputs - np.max(inputs,axis=1)[:,np.newaxis])
        return exp_inputs / np.sum(exp_inputs, axis=1)[:, np.newaxis]
    def weight_matrix_multiply(self,inputs,weights,biases): #completed, expensive, should be compiled
        #print "input dims are ", inputs.shape
        #print "weight dims are ", weights.shape
        #print "bias dims are ", biases.shape
        #return np.dot(inputs,weights)+np.tile(biases, (inputs.shape[0],1))
#        return slb.cblas.dgemm(alpha=1.0, a=inputs, b=weights) + biases
        out = np.dot(inputs,weights)
        out += biases
        return out
#        return np.dot(inputs,weights) + biases#[np.newaxis, :]