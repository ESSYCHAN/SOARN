import numpy as np
from collections import defaultdict, OrderedDict
from typing import List, Dict, Set, Any

import config
import element


class Stimulus:
    def __init__(self, group, symbol: str, trials: int = None, total_stimuli: int = None, corre_config: np.array = None):
        self.group = group
        self.symbol = symbol
        # self.beta = None
        # self.salience = None
        # print('sysmbol', symbol)
        self.trials = trials
        # print('TRILALSSSSSS', trials)
        self.total_stimuli = total_stimuli
        self.cues = None
        self.is_us = config.USNames.is_us(symbol)
        self.is_context = config.ContextConfig.isContext(symbol)
        self.is_cs = not self.is_us and not self.is_context
        self.max_duration = 1
        self.common_map = {}
        self.max_iti = 0
        self.disabled = False
        self.last_onset = 0
        self.last_offset = 1
        # self.current_phase = 0
        self.micros_set = False
        self.presence_mean = False
        self.presence_trace = 0
        self.presence_max = 1
        self.trial_types = []
        self.trial_count = 0
        self.onset = 0
        self.offset = 0
        # find a better way of setting tau
        self.is_probe = False
        self.random_trial_count = 0
        self.tau1 = 0.9
        self.tau2 = 0.87
        self.names = []
        # print(self.names)
        # if config.USNames.is_us(self.get_name()):
        #     self.names.append(self.get_name())
        self.ctx_fix = 1 if config.ContextConfig.isContext(
            self.get_name()) else 0
        self.presence_values = []
        self.presence_by_trial = [[]]
        self.associates = np.zeros((trials, total_stimuli), dtype=bool)
        self.temp_values = np.zeros(total_stimuli)
        self.temp_values_pred = np.zeros(total_stimuli)
        self.average_prediction = np.zeros(trials)
        self.complete_duration_list = []
        # self.was_active_last = np.zeros(
        #     group.get_no_of_phases(), dtype=bool)  # 1D
        # Nashuku its the same thing
        # print('BROOOO', group.get_no_of_phases())
        # self.activity = [None] * group.get_no_of_phases()
        # self.trial_string_activity = [None] * group.get_no_of_phases()
        # self.normal_activity = [None] * group.get_no_of_phases()
        self.average_average_weights = [0.0] * self.total_stimuli
        self.average_average_weights_a = [0.0] * self.total_stimuli
        self.raw_data = corre_config

        self.size = 0
        for names in self.symbol:
            # print(names)
            if names == 'c':  # know how to handle this
                pass
            elif names == '\u03A9':
                self.size = 1
            elif names == '-':
                self.size += self.raw_data['+'].img.filter_maps.size
            else:
                self.size += self.raw_data[names].img.filter_maps.size
        # print(self.size)
        # self.initialize_trial_arrays()

    def get(self, index):
        return self.cues[index]

    def get_name(self):
        return self.symbol

    def get_names(self):
        return self.names

    def get_symbol(self):
        return self.symbol

    def is_common(self):
        return len(self.get_name()) > 1 and self.get_name()[0] == 'c'

    def add_common(self, second: str, stimulus: 'Stimulus'):
        # c = {}
        self.common_map[second] = stimulus
        # c[str(self.group.get_name_of_grp())] = self.common_map
        # self.group.get_model().setCommonMap(common=c)

    def get_common_names(self) -> Set[str]:
        return set(self.common_map.keys())

    def get_common_map(self) -> Dict[str, 'Stimulus']:
        return self.common_map

    def get_the_onset(self) -> int:
        return self.onset

    def get_the_offset(self) -> int:
        return self.offset

    def get_average_weights_a(self) -> np.ndarray:
        return self.average_average_weights_a

    def set_timing(self, onset2: int, offset: int):
        self.onset = onset2
        self.offset = offset

    def get_max_duration(self) -> float:
        return self.max_duration

    def set_max_duration(self, new_max: int):
        # print('Executing this ', new_max)
        self.max_duration = new_max
        # self.set_all_max_duration(new_max)

    def set_all_max_duration(self, new_max: int):
        # print('executing setting all', self.group.get_phases())
        self.total_max = new_max

    def post_phase_init(self):
        for sp in self.group.get_phases():
            # print(sp.get_ITI().get_minimum())
            self.max_iti = max(self.max_iti, sp.get_ITI().get_minimum())
            # print('itiiiii', self.max_iti)
        self.delta_ws = np.zeros((self.group.get_no_of_phases(
        ), self.trials, self.total_max + int(self.max_iti)))
        self.asymptotes = np.zeros((self.group.get_no_of_phases(
        ), self.trials, self.total_max + int(self.max_iti)))

    def get_all_maxduration(self):
        return self.total_max
    # def __str__(self) -> str:
    #     return f"{self.symbol}{self.a_onset, self.a_offset}"

    def get_activity(self, phase: int) -> np.ndarray:
        return self.activity[phase]

    def get_trial_string_activity(self, phase: int) -> np.ndarray:
        return self.trial_string_activity[phase]

    def get_normal_activity(self, phase: int) -> np.ndarray:
        return self.normal_activity[phase]

    def set_trial_length(self, trial_length: int):
        self.trial_length = trial_length
        self.predictions_this_trial = [0]*trial_length
        self.errors_this_trial = [0] * trial_length

        for elemz in self.get_list():
            elemz.set_trial_length(trial_length)

    def set_activity(self, a: np.ndarray, phase: int):
        self.activity[phase] = np.copy(a)

    def set_trial_string_activity(self, a: np.ndarray, phase: int):
        self.trial_string_activity[phase] = np.copy(a)

    def set_normal_activity(self, a: np.ndarray, phase: int):
        self.normal_activity[phase] = np.copy(a)

    # def set_all_max_duration(self, new_max: int):
    #     self.total_max = new_max
    #     for sp in self.group.get_phases():
    #         self.max_iti = max(self.max_iti, sp.get_iti().get_minimum())
    #     # self.delta_ws = np.zeros((self.group.get_no_of_phases(
    #     # ), self.trials, self.total_max + int(self.max_iti)))
    #     # self.asymptotes = np.zeros((self.group.get_no_of_phases(
    #     # ), self.trials, self.total_max + int(self.max_iti)))

    def get_all_max_duration(self) -> int:
        return self.total_max

    def set_reset_context(self, reset: bool):  # clarify the function of this
        self.reset_context = reset

    def get_list(self):
        return self.cues

    def get_last_onset(self) -> int:
        return self.last_onset

    def add_microstimuli(self):
        # print('srim micro stimulu fxn', self.max_duration)

        if self.max_duration <= 0:
            self.max_duration = 1

        self.cues = [None] * int(self.max_duration) if not self.disabled else 1
        # print(self.cues)
        if not self.micros_set:
            for i in range(int(self.max_duration)if not self.disabled else 1):
                self.cues[i] = element.Element(microIndex=i, parent=self, group=self.group, name=self.get_name(),
                                               std=self.std, trials=self.trials, totalStimuli=self.total_stimuli,
                                               totalMax=int(self.total_max), vartheta=self.vartheta, presenceMean=self.presence_mean)

        self.micros_set = True
        for se in self.cues:
            # print(se.get_name())
            if self.is_context:
                se.setRAlpha(alpha=self.alphaR)
                se.setNAlpha(alphaN=self.alphaN)
            elif len(self.get_name()) > 1:
                se.setRAlpha(alpha=self.alphaR)
                se.setNAlpha(alphaN=self.alphaN)
                # se.setNAlpha(alphaN=self.alphaN)
            # We set the parameters of the context when setting the phase
            if self.is_us:
                # print(self.symbol)
                # print('wink', self.beta)
                # se.setBeta(ba=self.beta)
                se.set_CSC_like(c=self.cs_c_like)
                se.set_disabled(disbaled=False)
                se.setRAlpha(alpha=self.alphaR)
                se.setSalience(salience=self.salience)
                # se.setIntensity(f=self.i)
            else:
                se.set_CSC_like(c=self.cs_c_like)
                se.set_disabled(disbaled=False)
                se.setRAlpha(alpha=self.alphaR)
                se.setNAlpha(alphaN=self.alphaN)
                # se.setBeta(beta=self.beta)
                se.setSalience(salience=self.salience)

                # se.set_us_salience(self.omicron)

    # def reset(self, last: bool, current_trials: int):
    #     pass  # ToDo

    def reset_activation(self, context: bool):
        self.has_been_active = False
        if context and self.reset_context:  # Reset_context is a boolean value indisctiing if to reset or no
            self.start_decay = False
            self.presence_max *= self.reset_context
        self.presence_trace = 0
        self.duration_point = 0

        for cues in self.cues:
            cues.reset_activation()

    # Parameters of the stim from the model--->

    def set_r_alpha(self, alpha: float):
        if hasattr(self, "alphaR"):
            self.alphaR = alpha
        else:
            self.alphaR = alpha

    def set_n_alpha(self, alpha: float):
        if hasattr(self, "alphaN"):
            self.alphaN = alpha
        else:
            self.alphaN = alpha

    @property
    def set_beta(self):
        return self.beta

    @set_beta.setter
    def set_beta(self, beta: float):
        self.beta = beta

    @property
    def set_salience(self):
        return self.salience

    @set_salience.setter
    def set_salience(self, sal):
        self.salience = sal
        # if self.cues is not None:
        #     for se in self.cues:
        #         se.setSalience(
        #             sal=sal/self.max_duration if self.is_us else sal/self.max_duration)

    @property
    def set_omicron(self):
        return self.omicron

    @set_omicron.setter
    def set_omicron(self, o: float):
        # if not hasattr(self, 'omicron'):
        self.omicron = o

        # if self.cues is not None:
        #     for se in self.cues:
        #         se.set_us_salience(o) #Check alternative implelmentation on emelemts

    def set_b(self, b: float):  # bbackward discount
        self.b = b

    def set_csc_like(self, b: float):
        self.cs_c_like = b

    def get_b(self) -> float:
        return self.b

    def get_csc_like(self):
        return self.cs_c_like

    def get_r_alpha(self):
        return self.alphaR

    def get_n_alpha(self):
        return self.alphaN

    def get_salience(self):
        return self.salience

    def get_has_been_active(self):
        return self.has_been_active

    def set_parameters(self, std: float, vartheta: float):
        self.vartheta = vartheta
        self.std = std
        if self.cues is not None:
            for se in self.cues:
                se.set_vartheta(vartheta)

    def initialize(self, a, b):
        self.a_stim = a
        self.b_stim = b
        self.alphaR = (a.get_r_alpha() + b.get_r_alpha()) / 2.0
        self.alphaN = (a.get_n_alpha() + b.get_n_alpha()) / 2.0
        self.salience = (a.get_salience() + b.get_salience()) / 2.0
        for se in self.cues:
            se.initialize(a, b)
        self.curr_phase.presentStimuli.append(self.a_stim)
        self.curr_phase.presentStimuli.append(self.b_stim)

    # discovered after microsstim

    def setsize(self):
        # print(self.size, self.symbol)
        for se in self.cues:
            # print(self.cues)
            se.set_sub_set_size(self.size)
            # se.set_sub_set_size(10)

    def set_disabled(self):
        self.disabled = True
        self.alphaR = 0.5
        self.max_duration = 1
        self.last_onset = 0
        self.last_offset = 1
        self.current_phase = 0

    def set_presence_mean(self, b: bool):
        self.presence_mean = b

    def set_phase(self, phase):
        self.start_presence = self.presence_trace
        self.has_been_active = False
        self.combinations = 1
        self.trial_timepoints_counter = 0
        self.not_divided = True
        if self.reset_context:
            self.presence_max *= self.group.get_model().get_reset_value()  # reset value ya context
        self.curr_phase = self.group.get_phases()[phase]  # objjjj

        self.trial_type_map = OrderedDict()
        self.trial_type_map2 = OrderedDict()
        for trial_type in self.group.get_trial_strings():
            # print('yooo', trial_type)
            # Filtered basically is without the us
            filtered = trial_type[:-1] if config.USNames.has_us_symbol(
                trial_type) else trial_type
            # print(filtered)
            self.trial_type_map[trial_type] = 0
            self.trial_type_map2[filtered] = 0
        # print(config.ContextConfig().get_symbol())
        if self.curr_phase.get_context_config().get_symbol() == self.get_name():  # needs to set time before
            if hasattr(self, "is_context_values_set") and not self.is_context_values_set:
                self.is_context_values_set = True
            else:
                self.is_context_values_set = True

                for se in self.cues:
                    # print('hey', se)
                    if self.is_context:
                        self.alphaN = self.alphaN
                        self.alphaR = self.alphaR
                        # print(self.alphaN)
                        self.salience = self.salience
                        # print(self.salience)

                    se.setNAlpha(self.alphaN)
                    se.setRAlpha(self.alphaR)
                    # se.setSalience(self.salience)
                    se.setSalience(self.salience * 5.0 / self.max_duration)

        self.this_phase_max = self.presence_max
        self.trial_types.clear()
        self.current_phase = phase
        if phase == 0:

            if hasattr(self, "was_active_last") and self.was_active_last is not None:
                self.was_active_last[self.current_phase] = [False]
            else:
                self.was_active_last = [False] * self.group.get_no_of_phases()
######################
            if hasattr(self, "trial_string_activity") and self.trial_string_activity is not None:
                self.trial_string_activity[self.current_phase] = [
                    False]*(self.trials + 1)
            else:
                self.trial_string_activity = [
                    [False] * (self.trials + 1) for _ in range(self.group.get_no_of_phases())]
#####################
            if hasattr(self, "activity") and self.activity is not None:
                self.activity[self.current_phase] = [False]*(self.trials + 1)
            else:
                self.activity = [[False] * (self.trials + 1)
                                 for _ in range(self.group.get_no_of_phases())]
######################

            if hasattr(self, "normal_activity") and self.normal_activity is not None:
                self.normal_activity[self.current_phase] = [
                    False]*(self.trials + 1)
            else:
                self.normal_activity = [
                    [False] * (self.trials + 1) for _ in range(self.group.get_no_of_phases())]
#######################

        if phase > 0:
            self.was_active_last[self.current_phase] = self.was_active_last[self.current_phase - 1]

        self.average_average_weights = [0.0] * self.total_stimuli
        # print('stim_name iwant to see', self.get_name())

        for cue in self.cues:

            cue.set_phase(phase)

    def incremeant_combination(self):
        self.combinations += 1
        self.trial_types.clear()
        self.presence_max = float(self.this_phase_max)
        self.presence_trace = self.start_presence
        self.random_trial_count = 0
        self.trial_type_map = OrderedDict()
        self.trial_type_map2 = OrderedDict()

        # print('can i see this ', self.group.get_trial_strings())

        for s in self.group.get_trial_strings():
            # print('sssss', s)
            filtered = s[:-1] if config.USNames.has_us_symbol(names=s) else s
            # print('filtered', filtered)
            self.trial_type_map[s] = 0
            self.trial_type_map2[filtered] = 0

        for se in self.cues:
            se.next_combination()

    def reset(self, last: bool, current_trials: int):
        self.timepoint = 0
        if self.cues is not None:  # if they exist
            for cue in self.cues:  # element
                cue.reset(last, current_trials)
        self.has_been_active = False
        if not self.is_context:
            self.start_decay = False
        self.presence_trace = 0
        self.average_average_weights = [0.0] * self.total_stimuli
        self.average_error = [0.0] * (self.trials + 1)

        if hasattr(self, "activity") and self.activity is not None:
            self.activity[self.current_phase] = [False]*(self.trials + 1)
        else:
            self.activity = [[False] * (self.trials + 1)
                             for _ in range(self.group.get_no_of_phases())]

        if hasattr(self, "trial_string_activity") and self.trial_string_activity is not None:
            self.trial_string_activity[self.current_phase] = [
                False]*(self.trials + 1)
        else:
            self.trial_string_activity = [
                [False] * (self.trials + 1) for _ in range(self.group.get_no_of_phases())]

        self.temp_values = [0.0] * self.total_stimuli
        self.temp_values_pred = [0.0] * self.total_stimuli
        self.duration_point = 0
        # print('TRIAL before OUT', self.trial_count)
        self.trial_count -= min(self.trial_count, current_trials)
        # print('TRIAL SOUT', self.trial_count)

    def reset_completely(self):
        if self.cues is not None:
            for se in self.cues:
                se.reset_completley()
        self.trial_count = 0
        self.random_trial_wa = np.zeros(
            (self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.random_trial_wa_unique = np.zeros(
            (self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.trial_w = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.org_trial_w = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.trial_wa_compounds = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.trial_wa = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.org_trial_wa = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.average_average_weights = np.zeros(self.total_stimuli)
        self.average_average_weights_a = np.zeros(self.total_stimuli)
        self.average_prediction = np.zeros(self.trials)
        self.average_error = np.zeros(self.trials)
        self.duration_list = []
        self.complete_duration_list = []
        self.names = []
        self.presence_values = []
        self.presence_by_trial = [[]]
        self.temp_values = np.zeros(self.total_stimuli)
        self.temp_values_pred = np.zeros(self.total_stimuli)
        self.was_active_last = np.zeros(
            self.group.get_no_of_phases(), dtype=bool)
        self.associates = np.zeros(
            (self.trials, self.total_stimuli), dtype=bool)
        self.trial_types = []
        self.ctx_fix = 1 if config.ContextConfig.isContext(
            self.get_name()) else 0
        if config.USNames.is_us(self.symbol):
            self.is_us = True
        if config.ContextConfig.isContext(self.symbol):
            self.is_context = True

    def is_active(self):
        return self.active if not self.disabled else False

    # def update(self, first_pass: bool):
    #     pass

    def get_should_update(self) -> bool:
        return self.update

    def get_b(self) -> float:
        return self.b

    # def set_presence_trace(self, v):
    #     self.presence_trace = v

    # def get_presence_trace(self):
    #     self.presence_trace

    def update_presence_trace(self, duration: float):
        self.current_duration = duration

        if self.presence_trace < 0.01 and self.timepoint == 0:
            self.start_decay = False

        if self.is_active():
            self.has_been_active = True

        if self.is_active() and not self.start_decay:
            # print('TURE')
            self.presence_trace = 1.0
            self.start_decay = True

        elif self.is_active() and self.start_decay:
            self.presence_trace *= self.tau1

        elif not self.is_active() and self.has_been_active and self.start_decay:
            # print('yo')
            self.presence_trace *= self.tau2
            self.onset = 0

        if not self.is_active() and not self.has_been_active:
            self.presence_trace = 0.0

        if self.presence_trace > 1:
            self.presence_trace = 1

    def set_duration(self, onset, offset, duration_point, active, realtime):
        self.iti = self.curr_phase.get_current_iti()
        self.duration_point = duration_point-1
        # print('firest duration point', self.duration_point)
        self.active = active
        if offset > 0:
            self.last_offset = offset
        if onset != -1:
            self.last_onset = onset  # Start updating end of the time point
        if self.is_cs and realtime > self.last_onset+1 and realtime <= self.last_offset+2:
            # print(True, self.symbol)
            self.update = True
        elif self.is_us and realtime > self.last_onset+1 and realtime <= self.last_offset+2:
            # print(True, self.symbol)
            self.update = True
        elif not self.is_context:
            self.update = False
        elif self.is_context and realtime > self.last_onset+1 and realtime <= self.last_offset + self.iti + 1:
            self.update = True
        elif self.is_context:
            self.update = False

        self.update_presence_trace(duration=(offset-onset))
        # print('after update', self.presence_trace)

        self.update_duration = self.max_duration if self.current_duration == 0 else self.current_duration
        # print('update_duration', self.update_duration)

        for elem in self.get_list():
            elem.set_active(self.get_name(), active, duration_point)

        for n, ele in enumerate(self.get_list()):
            activation = self.presence_trace * \
                self.presence_max if n <= self.update_duration else 0

            # print('actiuvation state', self.presence_trace, activation)
            ele.update_activation(
                self.get_name(), activation, self.current_duration, n)

    def get_was_active(self) -> int:
        return 1 if (self.was_active_last[self.current_phase] and not self.has_been_active) else (1 if self.has_been_active else 0)

    def get_was_active_last(self) -> bool:
        return self.was_active_last[self.current_phase] and not self.has_been_active

    def get_prediction(self, name: str) -> float:
        if name in self.names:
            return self.average_weights_a[self.names.index(name)]
        else:
            return 0.0

    def get_v_value(self, name: str) -> float:
        if name in self.names:
            return self.average_weights[self.names.index(name)]
        else:
            return 0.0

    def increment_timepoint(self, time, iti):
        # print("====>>>>>>>", self.get_name(), self.names, self.has_been_active)
        temp_old_v = np.zeros(self.total_stimuli, dtype=float)
        if hasattr(self, "average_weights") and self.average_weights is not None and ("+" in self.names):
            old_v = self.average_weights[self.names.index("+")]
            # old_v = self.average_weights
        else:
            old_v = 0
        # old_v =  np.zeros(self.total_stimuli, dtype=float)
        # old_v = temp_old_v

        # old_v = self.average_weights[self.names.index(
        #     "+")] if self.average_weights is not None and ("+" in self.names or "-" in self.names) else 0
        # print('olllld', old_v)

        self.average_weights = np.zeros(self.total_stimuli, dtype=float)
        # print(self.average_weights)
        self.average_weights_a = np.zeros(self.total_stimuli, dtype=float)
        temp_map = np.zeros((self.total_stimuli, self.total_max), dtype=float)
        # print(temp_map)
        temp_map_pred = np.zeros(
            (self.total_stimuli, self.total_max), dtype=float)
        # self.average_weights = [0.0] * self.total_stimuli
        # self.average_weights_a = [0.0] * self.total_stimuli
        # temp_map = [[0.0] * self.total_max for _ in range(self.total_stimuli)]  # np.zeros((self.total_stimuli, self.total_max), dtype=float)
        self.delta_w = 0.0
        asymptote_mean = 0.0
        # print('before ', self.average_weights)

        for element in self.cues:
            # For the elements extract the weights with the key
            key_pred = element.increment_timepoint(time)[1]
            # print(key_pred)
            obj_pred = self.group.get_from_db(
                f"{self.current_phase}", key_pred)
            if obj_pred is not None:
                temp_map_pred = obj_pred
            # print(temp_map_pred)
            # print(yll.get())
            for i in range(self.total_stimuli):
                # temp_map = np.asarray(temp_map)
                # print('heYY',self.names)
                temp_w_pred = np.mean(temp_map[i])
                # temp_w_pred = np.sum(
                #     temp_map_pred[i]) / (len(self.names) if len(self.names) > i else 1)

                self.average_weights_a[i] += temp_w_pred

        for element in self.cues:
            # For the elements extract the weights with the key
            asymptote_mean += element.getAsymptote() / len(self.cues)
            key = element.increment_timepoint(time)[0]
            obj = self.group.get_from_db(f"{self.current_phase}", key)
            if obj is not None:
                temp_map = obj
            # print(temp_map)
            # print(yll.get())
            for i in range(self.total_stimuli):
                # temp_map = np.asarray(temp_map)
                # print('heYY',self.names)
                temp_w = np.mean(temp_map[i])
                # temp_w = np.sum(
                #     temp_map[i]) / (len(self.names) if len(self.names) > i else 1)

                # self.average_weights_a[i] += temp_w * \
                #     element.getGeneralActivation()
                self.average_weights[i] += temp_w

        # print('after', self.average_weights)

        # for x in range(self.total_stimuli):
        #     self.delta_w = self.average_weights[i]  - old_v[i]

        # print(self.delta_w)
        # print(halla.get())
        # print('----->',self.names)

        self.delta_w = self.average_weights[self.names.index(
            '+')] - old_v if ("+" in self.names or "-" in self.names) else 0 - old_v
        # print('ala', self.symbol, self.delta_w)
        # print(self.timepoint, self.trial_count)
        # print(self.not_divided)
        # self.delta_w = self.average_weights - old_v
        # print(self.delta_ws)
        self.delta_ws[self.current_phase][self.trial_count][self.timepoint] = self.delta_w
        # print(self.delta_ws)
        self.asymptotes[self.current_phase][self.trial_count][self.timepoint] = asymptote_mean
        if not self.has_been_active:
            self.duration_point = (time-1) - self.last_onset
            # self.duration_point = (time-1) - self.last_onset

        # print('last duration point', self.duration_point)

        if (self.not_divided and self.duration_point > 0 and
                self.duration_point - (self.last_offset - self.last_onset
                                       if self.has_been_active or self.is_context else self.max_duration) + self.ctx_fix * 2 <= 0):

            self.trial_timepoints_counter += 1
            self.average_average_weights += self.average_weights
            self.average_average_weights_a += self.average_weights_a
        if (self.not_divided and
                self.duration_point - (self.last_offset - self.last_onset if self.has_been_active or self.is_context else self.max_duration) == -2 * self.ctx_fix):
            divider = self.last_offset - \
                self.last_onset if self.has_been_active or self.is_context else self.max_duration
            # print(divider)
            self.average_average_weights /= divider
            self.average_average_weights_a /= divider
            self.not_divided = False
            # print('p2', self.symbol, self.average_weights)

        elif iti and self.not_divided:
            # print(True, '3')
            self.average_average_weights /= self.trial_timepoints_counter
            self.average_average_weights_a /= self.trial_timepoints_counter
            self.not_divided = False
            # print('p3', self.average_weights)

        self.combined_presence = self.presence_trace * self.presence_max
        self.presence_values.append(self.combined_presence)
        self.presence_by_trial[self.trial_count].append(self.combined_presence)
        if iti:
            self.trial_timepoints_counter = 0
        self.timepoint += 1
        # print(halla.het())

    def reset_for_next_timepoint(self):
        for element in self.cues:
            element.resetForNextTimepoint()

    def initialize_trial_arrays(self):
        self.trial_w = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.trial_wa_compounds = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.random_trial_wa = np.zeros(
            (self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.random_trial_wa_unique = np.zeros(
            (self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.org_trial_w = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.trial_wa = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))
        self.org_trial_wa = np.zeros((self.group.get_no_trial_types(
        ) + 1, self.group.get_no_of_phases(), self.trials, self.total_stimuli))

    def get_average_weights_a(self) -> np.ndarray:
        return self.average_weights_a

    def post_store(self):
        # pass
        self.average_average_weights = np.zeros(self.total_stimuli)
        self.average_average_weights_a = np.zeros(self.total_stimuli)

    def prestore(self):
        for name in self.names:
            if name != self.get_name():
                self.associates[self.trial_count][self.names.index(
                    name)] = self.has_been_active and self.group.get_cue_map().get(name).get_has_been_active()

        # print(self.associates)

    def store(self, trial_type: str):
        # print('TRAL TYPE', trial_type)
        self.not_divided = True
        self.trial_types.append(trial_type)
        self.was_active_last[self.current_phase] = self.has_been_active
        cur_t = self.curr_phase.get_current_trial()
        # print('current trial', cur_t)
        # print(self.symbol, self.average_average_weights)

        filtered = cur_t[:-
                         1] if config.USNames.has_us_symbol(names=cur_t) else cur_t
        # print('filtered trials', filtered)
        val = self.trial_type_map[cur_t]
        val2 = self.trial_type_map2[filtered]
        # self.trial_type_map[cur_t] = val + 1
        # self.trial_type_map2[filtered] = val2 + 1
        # print(val, val2,  self.trial_type_map, self.trial_type_map2, len(self.cues), self.common_map)
        # print(cur_t, self.group.get_trial_type_index(cur_t), self.current_phase,  self.trial_w.shape)
        # print(self.curr_phase.isRandom())
        # print(self.common_map)
        # print(val)

        if len(self.cues) > 0:
            for i in range(self.total_stimuli):

                self.temp_values[i] = self.average_average_weights[i]
                self.temp_values_pred[i] = self.average_average_weights_a[i]
                # print(self.temp_values_pred)

                self.trial_w[self.group.get_trial_type_index(cur_t) + 1][self.current_phase][val][i] += self.temp_values_pred[i] / (
                    float(self.group.get_model().getCombinationNo()) if self.curr_phase.isRandom() else 1.0)

        # self.trial_w[0][self.current_phase][self.trial_count] = self.temp_values.copy()
        # print(self.temp_values)
        # print(hala.get())
        # if len(self.cues) > 0:
        #     for i in range(self.total_stimuli):
        #         self.temp_values[i] = self.average_average_weights_a[i]
        #         # print(self.group.get_trial_type_index(cur_t), self.trial_wa.shape )
        #         # print(val)
        #         # common_wa = sum(common.get_average_weights_a()[i] / (float(self.group.get_model().getCombinationNo()) if self.curr_phase.isRandom() else 1.0) for common in self.common_map.values())
        #         # print(common_wa, self.get_name())
        #         # # print(halla.get())
        #         # update = False
        #         # if self.is_common():
        #         #     if self.get_name()[1] in cur_t or self.get_name()[2] in cur_t:
        #         #         update = True
        #         # else:
        #         #     print(self.get_name() in cur_t or self.is_context, self.get_name(), cur_t)
        #         #     if self.get_name() in cur_t or self.is_context:
        #         #         update = True
        #         #     elif self.get_name() == '+' and cur_t[-1] == '-':
        #         #         update = True

        #         # if update:
        #         #     self.random_trial_wa[self.current_phase][self.random_trial_count][i] += common_wa + (self.temp_values[i] / (float(self.group.get_model().getCombinationNo()) if self.curr_phase.isRandom() else 1.0))
        #         #     self.random_trial_wa_unique[self.current_phase][self.random_trial_count][i] += (self.temp_values[i] / (float(self.group.get_model().getCombinationNo()) if self.curr_phase.isRandom() else 1.0))

        #         self.trial_wa[self.group.get_trial_type_index(cur_t) + 1][self.current_phase][val][i] += self.temp_values[i] / (
        #             float(self.group.get_model().getCombinationNo()) if self.curr_phase.isRandom() else 1.0)
        #         self.trial_wa_compounds[self.group.get_trial_type_index2(filtered) + 1][self.current_phase][val2][i] += self.temp_values[i] / (
        #             float(self.group.get_model().getCombinationNo()) if self.curr_phase.isRandom() else 1.0)

        # print(self.temp_values)
        # the zeroth
        self.trial_wa[0][self.current_phase][self.trial_count] = self.temp_values.copy()
        self.trial_type_map[cur_t] = val + 1
        self.trial_type_map2[filtered] = val2 + 1

        self.prediction_sum = sum(self.predictions_this_trial)
        self.error_sum = sum(self.errors_this_trial)
        self.average_prediction[self.trial_count] = self.prediction_sum / \
            self.trial_length
        # print(self.average_prediction)
        self.average_error[self.trial_count] = self.error_sum / \
            self.trial_length
        self.activity[self.current_phase][self.trial_count] = self.has_been_active

        if self.is_probe:
            self.duration_list.append(int(self.current_duration))
        self.complete_duration_list.append(int(self.current_duration))

        for cue in self.cues:
            cue.store()

        self.trial_count += 1
        if self.is_common():
            if self.get_name()[1] in self.curr_phase.get_current_trial() or self.get_name()[2] in self.curr_phase.get_current_trial():
                self.random_trial_count += 1
        else:
            if self.get_name() in self.curr_phase.get_current_trial() or self.is_context:
                self.random_trial_count += 1

        self.presence_by_trial.append([])
        self.timepoint = 0

        if not self.is_context or self.reset_context == 1:
            self.has_been_active = False
            self.start_decay = False
            self.presence_max *= self.group.get_model().get_reset_value()
            self.presence_trace = 0

        self.average_average_weights = np.zeros(self.total_stimuli)
        # print('hey', self.symbol, self.average_prediction)

    def get_trial_average_weights(self, trial_type: int, phase: int) -> np.ndarray:
        # np.insert(self.trial_w, 0, np.zeros_like(
        #     self.trial_w[:, :, 1]), axis=0)
        return self.trial_w[trial_type][phase]

    def get_trial_average_weights_a(self, trial_type: int, phase: int) -> np.ndarray:
        # np.insert(self.trial_wa, 0, np.zeros_like(
        #     self.trial_wa[:, :, 1]), axis=0)
        return self.trial_wa[trial_type][phase]

    def get_avg_pred(self):
        return self.average_prediction

    def get_finale_wts(self):
        element_dict = {}
        for elemz in range(len(self.cues)):
            element_dict[elemz] = self.cues[elemz].sub_E_wts
        return element_dict
