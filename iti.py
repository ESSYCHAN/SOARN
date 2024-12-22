import time
import os
import math
import random
import itertools
from collections import defaultdict
import config


class ITI:
    def __init__(self, duration):
        self.duration = duration

    def wait(self):
        # print(f"Waiting for ITI of {self.duration} seconds...")
        time.sleep(self.duration)


class TimingConfiguration:
    def __init__(self, **kwargs):
        self.trial = 0
        self.onsets = defaultdict(int)
        # self.cs_Config = cs_config
        # print(self.cs_Config)
        self.sim_raww = {}
        for key, value in kwargs.items():
            self.sim_raww[value.get_symbol()] = value
        # print('time', self.sim_raww)

    def set_trials(self, trial):
        self.trial = trial

    def restart_onsets(self):
        self.onsets = defaultdict(int)
        # for tris in range(self.trials):
        #     self.onsets[tris] = 0

    # def trial_L(self, trial):
    #     trial_len = 0
    #     for cues in trial:

    def make_timings(self, cues):  # Cues from the trials  -  used CS config
        # print(cues)
        # print('aye', self.sim_raww)
        timings = {}

        for cue in cues:
            # print('==>', cue)
            # print(self.cs_Config[cue.get_symbol()])
            # if cue.get_symbol() == '\u03A9':
            #     timings[cue.get_symbol()] = [config.CS.OMEGA[cue.get_symbol()]
            #                                  [1], config.CS.OMEGA[cue.get_symbol()][0]]

            # elif cue.get_symbol() in ["+", "-", "0"]:
            #     timings[cue.get_symbol()] = [config.CS.US[cue.get_symbol()]
            #                                  [1], config.CS.US[cue.get_symbol()][0]]

            # elif not config.USNames.is_us(cue.get_symbol()) and not config.ContextConfig.isContext(cue.get_symbol()):
            #     timings[cue.get_symbol()] = config.CS.TOTAL[cue.get_symbol()
            #                                                 ][1], config.CS.TOTAL[cue.get_symbol()][0]

            # timings[cue.get_symbol()] = [cue[0], cue[1]]
            timings[cue.get_symbol()] = [self.sim_raww[cue.get_symbol()]
                                         [0], self.sim_raww[cue.get_symbol()][1]]
        return timings

    def t_max(self):
        # The higest value  offset value  of all cues in that trial
        total_max = 0
        for cue_key, cue_values in self.sim_raww.items():
            total_max = max(total_max, cue_values[1])
        # print('totalmax',total_max)
        return total_max

    def US_timings(self):
        timings = {}
        for cue_keys, cue_values in self.sim_raww.items():
            # print(cue_keys)
            if config.USNames.is_us(name=cue_keys):
                # print("TRUE", cue_keys)
                timings[cue_keys] = [cue_values[0], cue_values[1]]

        timings['-'] = [0, 0]
        timings['0'] = [cue_values[0], cue_values[1]]
        return timings

    def sequences(self, order):
        return order

    def compounds(self, order):
        comp = []
        for cues in order:
            y = cues.get_cues()

        return []


class ITIConfig:
    def __init__(self):
        self.minimum = 1

    def get_minimum(self):
        return self.minimum

    def reset(self):
        self.minimum = 1
