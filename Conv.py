from __future__ import annotations
from typing import Tuple, Optional
from matplotlib.pyplot import axis
from PIL import Image
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt


# https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast
# https://github.com/3outeille/CNNumpy/blob/master/src/slow/utils.py


class ConvLayer:

    # def __init__(self, w: np.array, padd: str = 'valid', stride: int = 1) -> None:

    #     # Convolution
    #     self._w = w
    #     self._padding = padd
    #     self._stride = stride
    #     self._a_prev = None

    # @classmethod
    # def initialize(cls, filters: int, kernel_shape: Tuple[int, int, int], padding: str = 'valid', stride: int = 1) -> ConvLayer:
    #     ww = np.random.randn(filters, *kernel_shape)*0.01
    #     # ww = ConvLayer.random_filtors(filters, kernel_shape)
    #     # print('ey', ww.shape)
    #     return cls(w=ww, padd=padding, stride=stride)

    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self._padding = padding
        self._stride = stride

        self._w = np.random.randn(self.n_F, self.n_C, self.f, self.f) 


    @property
    def weights(self):
        return self._w

    @property
    def data(self):
        return self.all_stim

    def forward_pass(self, a_prev: np.array) -> np.array:
        # self.cache = a_prev

        self._a_prev = np.array(a_prev, copy=True)

        ni, c, h_in, w_in = self._a_prev.shape
        n_C = self.n_F
        output_shape = self.calculate_output_dim(inp=self._a_prev)

        _, c_f, h_f, w_f = self._w.shape

        _, _, h_out, w_out = output_shape
       
        padd = self.calculate_pad_dims()

        a_prev_pad = self.padd(arraay=self._a_prev,  pad=padd)

        output = np.zeros((ni, n_C, h_out, w_out))

        for n_i in range(ni):
            for ch in range(n_C):
                for i in range(h_out):
                    h_start = i*self._stride
                    h_end = h_start+h_f
                    for j in range(w_out):

                        w_start = j*self._stride
                        w_end = w_start+w_f

                        # print(a_prev_pad[n_i, :, h_start:h_end, w_start:w_end].shape)
                        output[n_i, ch, i, j] = np.sum(
                            a_prev_pad[n_i, :, h_start:h_end, w_start:w_end] * self._w[ch, :,:,: ])/255


        return output

    def calculate_output_dim(self, inp):
        n, c, h_in, w_in = inp.shape
        n_f, _, h_f, w_f = self._w.shape

        if self._padding == 'same':
            return n, n_f, h_in, w_in

        elif self._padding == 'valid':

            h_out = (h_in-h_f)//self._stride+1
            w_out = (w_in-w_f)//self._stride+1

            return n, n_f, h_out, w_out

    # @staticmethod
    # def random_filtors(filt, kernels):
    #     j = np.zeros((filt, *kernels))
    #     # print(j.shape)
    #     for _num_fltrs in range(j.shape[0]):

    #         for row in range(len(j[_num_fltrs, :, :, :][0:])):
    #             for col in range(len(j[_num_fltrs, :, :, :][0])):

    #                 rr = np.random.randint(-5, 5)
    #                 j[_num_fltrs, :, row, col] = rr
    #     # print('aye', j)
    #     return j

    def calculate_pad_dims(self) -> Tuple[int, int]:
        if self._padding == 'same':
            # give me the same dimensons
            h_f, w_f, _, _ = self._w.shape
            return (h_f - 1)//2, (w_f - 1)//2
        elif self._padding == 'valid':
            return 0, 0

        else:
            pass

    @staticmethod
    def padd(arraay: np.array, pad: Tuple[int, int]) -> np.array:
        return np.pad(array=arraay, pad_width=((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), mode='constant')