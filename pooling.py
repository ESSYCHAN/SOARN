from __future__ import annotations
from typing import Tuple
from matplotlib.pyplot import axes
import numpy as np

from base import Layer


class Maxpool(Layer):
    def __init__(self, pool_size: Tuple[int, int], stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self.a = None
        self.cache = {}
        # print(self.cache)

    def forward_pass(self, a_prev: np.array) -> np.array:
        self.a = np.array(a_prev, copy=True)
        n, c, h_in, w_in = a_prev.shape
        h_pool, w_pool = self.pool_size

        h_out = 1 + (h_in - h_pool)//self.stride
        w_out = 1 + (w_in - w_pool)//self.stride

        output = np.zeros((n, c, h_out, w_out))
        for n_i in range(n):
            for ch in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        h_end = h_start + h_pool

                        w_start = j * self.stride
                        w_end = w_start + w_pool

                        output[n_i, ch, i, j] = np.max(
                            a_prev[n_i, ch, h_start:h_end, w_start: w_end])
                    # self.save_mask(x=a_prev_slice, cords=(i,j))

        return output

    def save_mask(self, x: np.array, cords: Tuple[int, int]) -> None:
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self.cache[cords] = mask
        # print(self.cache)
