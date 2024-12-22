from __future__ import annotations
from typing import List, Dict, Callable, Optional
import time
import matplotlib.pyplot as plt
import numpy as np
import Conv
import pooling
# from core import generate_batches, format_time
# from metrics import softmax_accuracy, softmax_cross_entropy



from base import Layer, Optimizer

# class SequentialModel:
#     def __init__(self, layers: List[Layer] ):
#         self._layers = layers 
#         # self._optimizer = optimizer 
#         # optimizer: Optimizer
#         self.layer_out = {}
#         self._train_acc = []
#         self._test_acc = []
#         self._train_loss = []
#         self._test_loss = []
        

#     def train(self, x_train: np.array) -> None:
        
#         y_hat_batch = self._forward(x_train)
#         return y_hat_batch

#             # print(y_hat_batch.shape)

#     def _forward(self, x: np.array) -> np.array:
#         activation = x

        
       
#         for idx, layer in enumerate(self._layers):
#             activation = layer.forward_pass( a_prev=activation)
   
#             self.layer_out[idx] = activation
            
#         return activation

#     @property
#     def outputs(self):
#         return self.layer_out

class SequentialModel:
    def __init__(self ):
        self.dur = 4
        self.conv1 = Conv.ConvLayer(nb_filters=self.dur, filter_size=3, nb_channels=3, stride=1, padding='valid')
        self.conv2 = Conv.ConvLayer(nb_filters=self.dur, filter_size=3, nb_channels=self.dur, stride=1, padding='valid')
        self.pool1 = pooling.Maxpool(pool_size=(4, 4), stride=2)

        self.conv3 = Conv.ConvLayer(nb_filters=self.dur, filter_size=3, nb_channels=self.dur, stride=1, padding='valid')
        self.conv4 = Conv.ConvLayer(nb_filters=self.dur, filter_size=3, nb_channels=self.dur, stride=1, padding='valid')
        self.conv5 = Conv.ConvLayer(nb_filters=self.dur, filter_size=3, nb_channels=self.dur, stride=1, padding='valid')

        # self.conv3 = Conv.ConvLayer.initialize(filters=self.dur, kernel_shape=(3, 3, 3), padding='valid', stride=1)
        # self.conv4 = Conv.ConvLayer.initialize(filters=self.dur, kernel_shape=(3, 3, 3), padding='valid', stride=1)
        # self.conv5 = Conv.ConvLayer.initialize(filters=self.dur, kernel_shape=(3, 3, 3), padding='valid', stride=1)

        self.pool2 =pooling.Maxpool(pool_size=(4, 4), stride=2)
        self.pool3 =pooling.Maxpool(pool_size=(2, 2), stride=1)
        self.pool4 =pooling.Maxpool(pool_size=(2, 2), stride=1)

        # self._optimizer = optimizer 
        # optimizer: Optimizer
        self.layer_out = {}
        self._train_acc = []
        self._test_acc = []
        self._train_loss = []
        self._test_loss = []
        

    def train(self, x_train: np.array) -> None:
        
        y_hat_batch = self._forward(x_train)
        return y_hat_batch

            # print(y_hat_batch.shape)

    def _forward(self, x: np.array) -> np.array:
        activation = x

        conv1 = self.conv1.forward_pass(a_prev=activation)
        conv2 = self.conv2.forward_pass(a_prev=conv1)
        
        pool1 = self.pool1.forward_pass(a_prev=conv2)

        conv3 = self.conv3.forward_pass(a_prev=pool1)
        conv4 = self.conv4.forward_pass(a_prev=conv3)
        conv5 = self.conv5.forward_pass(a_prev=conv4)

        pool2 = self.pool2.forward_pass(a_prev=conv5)
        pool3 = self.pool3.forward_pass(a_prev=pool2)
        pool4 = self.pool4.forward_pass(a_prev=pool3)


        
       
        # for idx, layer in enumerate(self._layers):
        #     activation = layer.forward_pass( a_prev=activation)
   
        #     self.layer_out[idx] = activation
            
        return pool4

    @property
    def outputs(self):
        return self.layer_out


