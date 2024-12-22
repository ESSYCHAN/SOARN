import numpy as np
# import Element
import utils
# import Conv
# import pooling
import Sequential
import matplotlib.pyplot as plt
import pandas as pd

import config
import element


# Set the size of the image
width = 4
height = 4
# Generate random pixel values for the image (range: 0 to 255)
random_image = np.random.randint(
    0, 256, size=(height, width, 3), dtype=np.uint8)
# print(random_image.size)


class Stim:
    def __init__(self, num_FM: int = 5, symbol: str = 'A',  **kwargs):
        self.kwargs = kwargs

        for key, value in kwargs.items():
            self.stim_lable = key
            self.stim_lnk = value
        self.dur = num_FM
        self.symbol = symbol
        self.raw_imges = utils.load_data(data_path=self.stim_lnk)
        # print(self.raw_imges.shape)

        self.model = Sequential.SequentialModel()

        self.filter_maps = self.process()
        self.status = [1]

        self.is_us = config.USNames.is_us(symbol)
        self.is_context = config.ContextConfig.isContext(symbol)
        self.is_cs = not self.is_us and not self.is_context
        self.combinations = 1
        # self.activation_cont = self._init_DA_ele()
        # self.parameters = parameters
        # # print(self.parameters.TOTAL)
        # if config.CS.isContext(name=self.symbol):
        #     self.duration = (
        #         self.parameters.OMEGA[self.symbol][1] - self.parameters.OMEGA[self.symbol][0])+1
        # elif config.CS.isUS(name=self.symbol):
        #     self.duration = (
        #         self.parameters.US[self.symbol][1] - self.parameters.US[self.symbol][0])+1
        # else:
        #     self.duration = (
        #         self.parameters.TOTAL[self.symbol][1] - self.parameters.TOTAL[self.symbol][0])+1
        # print(self.duration)
        self.alphaR = 0.1
        self.std = 1
        self.vartheta = 0.5

    def __add__(self, other):
        assert isinstance(other, type(self))
        comp = self.__class__()

    def __setitem__(self, key, value):
        if key == 0:
            self.status[key] = value
        else:
            raise ValueError({key}, 'needs to be 0 to change the status')

    # def set_parameter(self, key, value):
        # self.parameters[key] = value

    def set_parameter(self, beta, alpha_n, alpha_R, salience, std, vartheta):
        self.beta = beta
        self.alphaN = alpha_n
        self.alphaR = alpha_R
        self.salience = salience

        self.std = std
        self.vartheta = vartheta

    def get_parameter(self, key):
        return self.parameters.get(key, None)

    @property
    def stim_presence(self):
        return self.status[0]

    def Extract_OA(self, elemz):
        Act_cont = {}
        for key_stim, value_stim in self.FMS.items():
            Act_cont.setdefault(key_stim, {})
            for key_FM, value_FM in value_stim.items():
                Act_cont[key_stim][key_FM] = np.zeros_like(value_FM)

                for row in range(len(value_FM[0:])):
                    for col in range(len(value_FM[0])):
                        # print(value_FM[row][col])
                        # print(elemz[value_FM[row][col]].DA)

                        Act_cont[key_stim][key_FM][row][col] = elemz[value_FM[row][col]].DA

                # Act_cont[key_stim][key_FM]=pd.DataFrame(np.zeros(len(np.ravel(value_FM))), index=list(np.ravel(value_FM)))
        return Act_cont

    def process(self):
        # self.filter_maps = {}
        fmz = self.model.train(x_train=self.raw_imges)
        # print('fmz shape', fmz.shape)
        n, n_f, h_out, w_out = fmz.shape
        # print('zozz', fmz.shape)

        # for i in range(fmz.shape[2]):
        #     self.filter_maps[i] = Filter_map(on = self.on, off=self.end_time, typ=self.typ, raw_data= fmz[:,:, i], spec=self.spec, FM = i, FM_stim=self.stim_lable)
        # # print(self.filter_maps)
        new_fmz = fmz.reshape(n*n_f, h_out, w_out)
        return new_fmz

    @property
    def FMS(self):
        self.FM = {self.stim_lable: {}}
        for filter in range(self.filter_maps.shape[0]):
            self.FM[self.stim_lable][filter] = self.filter_maps[filter, :, :]
        return self.FM

    def Direct_Activation(self, elemz):
        DA_cont = {}

        for key_stim, val_stim in self.FMS.items():
            DA_cont.setdefault(key_stim, {})
            for key_FM, val_FM in val_stim.items():

                # select_ele = random.sample(population= list(np.ravel(val_FM)), k =random.randint(a =0, b =val_FM.size))
                # # print(select_ele)

                # mask
                DA_cont[key_stim][key_FM] = np.isin(val_FM, elemz).astype(int)

        return DA_cont

    def add_microstim(self):
        if self.duration <= 0:
            self.duration = 1

        self.cues = [None] * self.duration

        # print(self.cues)

        for i in range(self.duration):
            self.cues[i] = element.Element(index=i, parent=self, name=self.symbol,
                                           std=self.std, totalMax=int(self.duration), vartheta=self.vartheta)

        for se in self.cues:
            if self.is_context:
                pass
            se.setRAlpha(alpha=self.alphaR)
            se.setNAlpha(alphaN=self.alphaN)
            se.setBeta(beta=self.beta)
            se.setSalience(salience=self.salience)
