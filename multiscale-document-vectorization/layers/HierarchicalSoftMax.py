#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:22:21 2021

@author: peter
"""

import keras
import tensorflow
import numpy

class HierarchicalSoftMax(keras.layers.Layer):
    
    def __init__(self,n):#structure,row=-1,order=None):
        super(HierarchicalSoftMax,self).__init__()
        # self.structure = structure
        # self.row = row
        self.normal = None
        self.bias = self.add_weight(shape=(1,),
                                    initializer='random_normal',
                                    trainable=True)
        self.n_outputs = n
        # self.order = [] if order is None else order
        l = n//2
        if l==1:
            self.left=keras.layers.Lambda(lambda x: tensorflow.constant(1.0,
                                                                        shape=(1,)))
        else:
            self.left=HierarchicalSoftMax(l)
        if n-l==1:
            self.right=keras.layers.Lambda(lambda x: tensorflow.constant(1.0,
                                                                         shape=(1,)))
        else:
            self.right=HierarchicalSoftMax(n-l)
        self.concat = keras.layers.Concatenate()
            
        
    def build(self,input_shape):
        self.normal = self.add_weight(shape=(input_shape[-1],),
                                       initializer='he_normal',
                                       trainable=True)
        self.left.build(input_shape)
        self.right.build(input_shape)
        self.built = True
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(self.n_outputs,)
        
    def call(self,X,training=None):
        
        y=keras.activations.sigmoid(tensorflow.tensordot(X,
                                                         self.normal,
                                                         1)+self.bias)
        result = self.concat([y*self.left(X),(1.0-y)*self.right(X)])
        return result
        
    
    def get_config(self):
        return {'n':self.n_outputs}
    
    @classmethod
    def from_config(cls,config):
        return cls(config['n'])
        
    def embedding(self,shape,dtype=None):
        if not self.built:
            self.build(shape)
        weights = self.get_weights()
        vec = None
        for weight in weights:
            if weight.shape[0]>1:
                vec = weight
        l = self.n_outputs//2
        result = []
        if l == 1:
            result.append(vec)
        else:
            result.append.left.embedding() + vec
        if self.n_outputs - l == 1:
            result.append(-vec)
        else:
            result.append.right.embedding() - vec
        return numpy.vstack(result)
    