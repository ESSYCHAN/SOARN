import numpy as np
import pandas as pd


import Raw_stim
from collections import defaultdict, OrderedDict
import config
import random
import Phase

import math


class Element:

    def __init__(self, microIndex: int, parent, group, name: str,
                 std: float, trials: int, totalStimuli: int,
                 totalMax: int, vartheta: float, presenceMean: bool):

        self.microstimulusIndex = microIndex
        self.microPlus = microIndex+1
        self.totalTrials = trials + 1
        self.parent = parent
        self.group = group
        self.totalStimuli = totalStimuli
        self.totalMax = totalMax
        self.vartheta = vartheta
        self.presenceMean = presenceMean
        self.std = std
        self.name = name
        self.names = list()
        self.trialTypeCount = defaultdict(int)
        self.subsetSet = False  # initial value
        self.trialCount = 0
        # self.phase = 0  # hmm coz of set phase
        self.is_us = config.USNames.is_us(name)
        self.is_context = config.ContextConfig.isContext(name)
        self.is_cs = not self.is_us and not self.is_context
        # if config.USNames.is_us(self.get_name()):
        #     self.names.append(self.get_name())

        # self.isA = False
        # self.isB = False
        self.maxActivation = 0
        self.disabled = False
        self.intensity = 1  # find a better way of setting it
        self.cscLikeness = 1
        # self.activation = 0.0
        self.assoc = 0
        self.generalActivation = None
        self.factor = 10.0
        self.averageUSError = 0
        self.averageCSError = 0
        self.ownActivation = 0

        self.iti = max(sp.get_ITI().get_minimum() for sp in group.get_phases())
        self.presences = [0.0] * totalStimuli
        self.totalCSError = [0.0] * group.get_no_of_phases()
        self.totalUSError = [0.0] * group.get_no_of_phases()
        self.temp = [0.0] * (totalMax + self.iti)  # placeholder

        self.variableSalience = [0.0] * group.get_no_of_phases()  # Alpha
        self.csVariableSalience = [0.0] * \
            group.get_no_of_phases()  # Alpha of the CS
        # Keyssss
        self.elementDAKey = f"{group.get_name_of_grp()}{parent.get_name()}{self.microstimulusIndex}elementDirectActivity"
        self.elementCurrentWeightsKey = f"{group.get_name_of_grp()}{parent.get_name()}{self.microstimulusIndex}elementCurrentWeights"
        self.elementPredictionKey = f"{group.get_name_of_grp()}{parent.get_name()}{self.microstimulusIndex}elementPrediction"
        self.eligibilitiesKey = f"{group.get_name_of_grp()}{parent.get_name()}{self.microstimulusIndex}eligibilities"
        self.aggregateActivationsKey = f"{group.get_name_of_grp()}{parent.get_name()}{self.microstimulusIndex}aggregateActivations"
        self.aggregateSaliencesKey = f"{group.get_name_of_grp()}{parent.get_name()}{self.microstimulusIndex}aggregateSaliences"
        self.aggregateCSSaliencesKey = f"{group.get_name_of_grp()}{parent.get_name()}{self.microstimulusIndex}aggregateCSSaliences"

        group.make_map(self.elementCurrentWeightsKey)
        group.make_map(self.elementPredictionKey)
        group.make_map(self.eligibilitiesKey)
        group.make_map(self.aggregateActivationsKey)
        group.make_map(self.aggregateSaliencesKey)
        group.make_map(self.aggregateCSSaliencesKey)

        for i in range(group.get_no_of_phases()):
            group.add_to_map(key=str(i), entry=[[0.0] * totalMax for _ in range(totalStimuli)],
                             map_name=self.elementCurrentWeightsKey, add_now=True)

        for i in range(group.get_no_of_phases()):
            group.add_to_map(key=str(i), entry=[[0.0] * totalMax for _ in range(totalStimuli)],
                             map_name=self.elementPredictionKey, add_now=True)

        for i in range(totalStimuli):
            group.add_to_map(key=str(i), entry=[
                             0.0] * totalMax, map_name=self.eligibilitiesKey, add_now=True)

        for i in range(trials+1):
            group.add_to_map(key=str(i), entry=[
                             0.0]*(totalMax + self.iti), map_name=self.elementDAKey, add_now=True)
            group.add_to_map(key=str(i), entry=[
                             0.0]*(totalMax + self.iti), map_name=self.aggregateActivationsKey, add_now=True)
            group.add_to_map(key=str(i), entry=[
                             0.0]*(totalMax + self.iti), map_name=self.aggregateSaliencesKey, add_now=True)
            group.add_to_map(key=str(i), entry=[
                             0.0]*(totalMax + self.iti), map_name=self.aggregateCSSaliencesKey, add_now=True)

        # self.DA = [0] * self.t

        self.adj = 0 if self.is_us else 0
        # print(group.maps)

    # def se.set_n_alpha(self.alpha_n)
    # def se.set_r_alpha(self.alpha_r)
    # def se.set_salience(self.salience)
    # def se.set_beta(self.beta)\

    def get_names(self):
        return self.names

    # Called during setting up thr micro stimuli function but set_phase for context

    def setRAlpha(self, alpha: float):
        self.alphaR = alpha
        self.variableSalience[0] = alpha

    def setNAlpha(self, alphaN: float):
        self.alphaN = alphaN
        self.csVariableSalience[0] = alphaN

    def setBeta(self, ba: float):
        self.beta = ba

    def setSalience(self, salience: float):
        self.salience = salience

    def set_disabled(self, disbaled: bool):
        self.disabled = disbaled

    def set_CSC_like(self, c: float):
        self.cscLikeness = c

    def setIntensity(self, f: float):
        self.intensity = f

    def getSTD(self):
        return self.std

    def initialize(self, a, b):
        self.a = a
        self.b = b
        if hasattr(self, 'isA') and not self.isA:
            self.isA = True
        else:
            self.isA = True

        if hasattr(self, 'isB') and not self.isB:
            self.isB = True
        else:
            self.isB = True

        self.setParams()

    def setParams(self):

        if 'c' in self.get_name():
            if self.isA and self.isB and hasattr(self, 'notTotal') and not self.notTotal:
                self.std = (self.a.get_list()[
                            0].getSTD()+self.b.get_list()[0].getSTD())/2
                self.alphaR = (self.a.get_r_alpha() + self.b.get_r_alpha())/2
                self.alphaN = (self.a.get_n_alpha() + self.b.get_n_alpha())/2

                if self.trialCount == 0:
                    self.variableSalience[0] = self.alphaR
                    self.csVariableSalience[0] = self.alphaN
                self.variableSalience[self.phase] = self.alphaR
                self.csVariableSalience[self.phase] = self.alphaN
                self.salience = (self.a.get_salience() +
                                 self.b.get_salience())/2
                self.cscLikeness = (self.a.get_csc_like() +
                                    self.b.get_csc_like())/2
                self.averageUSError = max(0, self.alphaR)
                self.averageCSError = max(0, self.alphaN)
                self.notTotal = False
                self.kickedIn = True

            elif self.isA and hasattr(self, 'kickedIn') and not self.kickedIn:
                self.std = self.a.get_list()[0].getSTD()
                self.alphaR = self.a.get_r_alpha()
                self.alphaN = self.a.get_n_alpha()
                if self.trialCount == 0:
                    self.variableSalience[0] = self.alphaR
                    self.csVariableSalience[0] = self.alphaN
                self.averageUSError = max(0, self.alphaR)
                self.averageCSError = max(0, self.alphaN)
                self.salience = self.a.get_salience()
                self.cscLikeness = self.a.get_csc_like()
                self.kickedIn = True
            elif self.isB and hasattr(self, 'kickedIn') and not self.kickedIn:
                self.std = self.b.get_list()[0].getSTD()
                self.alphaR = self.b.get_r_alpha()
                self.alphaN = self.b.get_n_alpha()
                if self.trialCount == 0:
                    self.variableSalience[0] = self.alphaR
                    self.csVariableSalience[0] = self.alphaN
                self.averageUSError = max(0, self.alphaR)
                self.averageCSError = max(0, self.alphaN)
                self.salience = self.b.get_salience()
                self.cscLikeness = self.b.get_csc_like()
                self.kickedIn = True

    def get_name(self):
        return self.name

    def get_microIndex(self) -> int:
        return self.microstimulusIndex

    # Set in the Algorithim
    def set_trial_length(self, trial_len: int):
        self.microIndex = self.microPlus

        self.ctxratio = (trial_len/(len(self.parent.get_list())+0.0)
                         ) * (10.0/10.0) if self.is_context else 1.0
        self.ratio = (trial_len/(len(self.parent.get_list())+0.0)) * \
            (10.0/10.0) if self.is_context else 1.0

        self.denominator = math.pow((math.sqrt((self.microPlus+self.adj) * self.USCV * self.ctxratio)
                                    if self.is_us else math.sqrt((self.microPlus+self.adj) * self.std * self.ctxratio)), 2)
        # print('denominator', self.name,  self.denominator)

    # Set called by function in phase

    def set_CSCV(self, cscv: float):
        self.std = cscv

    def set_USCV(self, uscv: float):
        self.USCV = uscv

    def set_USScalar(self, usScalar: float):
        self.USCV *= (usScalar * usScalar)

    def set_CSScalar(self, csScalar: float):
        self.std *= (csScalar * csScalar)

    def set_USPersistence(self, p: float):
        self.usPersistence = p

    def setCurrentTrialString(self, currentString: str):
        if currentString in self.trialTypeCount:
            self.trialTypeCount[currentString] += 1
        else:
            self.trialTypeCount[currentString] = 1
        self.currentTrialString = currentString
        # print(self.trialTypeCount)
        # print(self.currentTrialString)

    def set_phase(self, phase: int):
        # print('SELN NMAME', self.get_name())

        self.session = 1
        self.combination = 1
        self.trialTypeCount.clear()

        # Set pahse
        self.phase = phase  # just a number

        if phase == 0:
            self.subelementWeights = [[[0.0] * self.totalMax for _ in range(
                self.totalStimuli)] for _ in range(int(self.subelementNumber))]
            # print(np.asarray(self.subelementWeights).shape)
            self.oldElementWeights = self.subelementWeights
            self.kickedIn = False
            # self.variableSalience[0] = self.alphaR
            # self.csVariableSalience[0] = self.alphaN
            self.totalCSError[phase] = 0
            self.totalUSError[phase] = 0
            self.group.add_to_map(key="0", entry=[[0.0] * self.totalMax for _ in range(
                self.totalStimuli)], map_name=self.elementCurrentWeightsKey, add_now=True)

            self.group.add_to_map(key="0", entry=[[0.0] * self.totalMax for _ in range(
                self.totalStimuli)], map_name=self.elementPredictionKey, add_now=True)
        # if phase> self.phase:
        #     for i in range(len(self.subelementWeights)):
        #         for j in range(len(self.subelementWeights[0])):
        #             for k in range(len(self.subelementWeights[0][0])):

        if phase > 0:
            self.totalCSError[phase] = self.totalCSError[phase-1]
            self.totalUSError[phase] = self.totalUSError[phase-1]
            self.variableSalience[phase] = self.variableSalience[phase-1]
            self.csVariableSalience[phase] = self.csVariableSalience[phase - 1]
            for j in range(len(self.subelementWeights[0])):  # totalnumerostim
                for k in range(len(self.subelementWeights[0][0])):  # totalMax
                    for i in range(len(self.subelementWeights)):  # subelementnumber
                        self.subelementWeights[i][j][k] = self.oldElementWeights[i][j][k]

            temp = self.group.get_from_db(
                key=str(phase-1), mapname=self.elementCurrentWeightsKey)
            if temp is not None:
                self.group.add_to_map(key=str(
                    phase-1), entry=temp, map_name=self.elementCurrentWeightsKey, add_now=True)

            else:
                temp = [
                    [0.0] * self.totalMax for _ in range(self.totalStimuli)]

                self.group.add_to_map(key=str(
                    phase-1), entry=temp, map_name=self.elementCurrentWeightsKey, add_now=True)

            temp2 = self.group.get_from_db(
                key=str(phase-1), mapname=self.elementPredictionKey)
            if temp2 is not None:
                self.group.add_to_map(key=str(
                    phase-1), entry=temp2, map_name=self.elementPredictionKey, add_now=True)

            else:
                temp2 = [
                    [0.0] * self.totalMax for _ in range(self.totalStimuli)]

                self.group.add_to_map(key=str(
                    phase-1), entry=temp2, map_name=self.elementPredictionKey, add_now=True)

        # if phase - 2 >= 0:
        #     self.group.get_maps().get(self.elementCurrentWeightsKey).remove(str(phase - 2))

    def set_sub_set_size(self, i: int):
        # print('=====>', self.get_name())
        if not self.subsetSet:
            self.totalElements = i
            self.discount = self.group.get_model().get_discount()

            self.dis = math.pow(self.discount, 0.01)
            self.exponent = 0 if self.dis == 0 else abs(1 / self.dis - 1)
            # print("i am exponent", self.exponent, self.discount, self.dis)

            self.outOfBounds = False
            if math.isnan(self.exponent) or self.exponent > 20:
                self.outOfBounds = True

            self.subsetSet = True
            isCommon = len(self.get_name()) > 1
            hasCommon = len(self.parent.get_common_map()) > 0
            commonProp = i * self.group.get_model().getCommon().get(self.group.get_name_of_grp()
                                                                    ).get(self.get_name()) if isCommon else 0

            # print('common prop', commonProp)
            uniqueProp = i * (1 - self.group.get_model().getProportions().get(
                self.group.get_name_of_grp()).get(self.get_name())) if hasCommon else 0

            # print('uniquep', uniqueProp)

            commonValue = commonProp if isCommon else (
                uniqueProp if hasCommon else i)
            # print('wjwjwkjw', commonValue)s
            self.subelementNumber = math.floor(commonValue)
            # print('SUBELEMEMT NUMERRROO',self.name, self.subelementNumber)
            self.subelementNumber = max(
                1, self.subelementNumber) if commonValue > 0 else self.subelementNumber

            # print(self.subelementNumber )
            if self.is_context:
                self.subelementNumber = 1
            # print('SUBELEMEMT NUMERRROO', self.name, self.subelementNumber)

            self.subelementWeights = [[[0.0] * self.totalMax for _ in range(
                self.totalStimuli)] for _ in range(int(self.subelementNumber))]
            self.oldElementWeights = [[[0.0] * self.totalMax for _ in range(
                self.totalStimuli)] for _ in range(int(self.subelementNumber))]

        self.subelementActivations = [0] * int(self.subelementNumber)

    def next_combination(self):
        self.combination += 1
        self.session = 1
        self.trialTypeCount.clear()

    def reset(self, last: bool, currentTrials: int):
        self.activation = 0.0
        self.assoc = 0
        if self.phase > 0:
            self.totalCSError[self.phase] = self.totalCSError[self.phase - 1]
            self.totalUSError[self.phase] = self.totalUSError[self.phase - 1]
            self.variableSalience[self.phase] = self.variableSalience[self.phase - 1]
            self.csVariableSalience[self.phase] = self.csVariableSalience[self.phase - 1]
            for j in range(len(self.subelementWeights[0])):
                for k in range(len(self.subelementWeights[0][0])):
                    for i in range(len(self.subelementWeights)):
                        self.subelementWeights[i][j][k] = self.oldElementWeights[i][j][k]

            current = self.group.get_from_db(
                str(self.phase - 1), self.elementCurrentWeightsKey)
            if current is not None:
                self.group.add_to_map(str(self.phase), current,
                                      self.elementCurrentWeightsKey, True)
            else:
                current = [
                    [0.0] * self.totalMax for _ in range(self.totalStimuli)]  # need to add sub element weights
                self.group.add_to_map(str(self.phase), current,
                                      self.elementCurrentWeightsKey, True)
            if self.phase - 2 >= 0:
                self.group.get_maps().get(self.elementCurrentWeightsKey).remove(str(self.phase - 2))

            current2 = self.group.get_from_db(
                str(self.phase - 1), self.elementPredictionKey)
            if current2 is not None:
                self.group.add_to_map(str(self.phase), current2,
                                      self.elementPredictionKey, True)
            else:
                current2 = [
                    [0.0] * self.totalMax for _ in range(self.totalStimuli)]  # need to add sub element weights
                self.group.add_to_map(str(self.phase), current2,
                                      self.elementPredictionKey, True)
            if self.phase - 2 >= 0:
                self.group.get_maps().get(self.elementPredictionKey).remove(str(self.phase - 2))

        if self.phase == 0:
            self.subelementWeights = [[[0.0] * self.totalMax for _ in range(
                self.totalStimuli)] for _ in range(int(self.subelementNumber))]
            self.group.add_to_map("0", [[0.0] * self.totalMax for _ in range(
                self.totalStimuli)], self.elementCurrentWeightsKey, True)
            self.group.add_to_map("0", [[0.0] * self.totalMax for _ in range(
                self.totalStimuli)], self.elementPredictionKey, True)
        if self.subelementActivations is not None:
            for i in range(len(self.subelementActivations)):
                self.subelementActivations[i] = 0
        for i in range(self.totalStimuli):
            self.group.add_to_map(
                str(i), [0.0] * self.totalMax, self.eligibilitiesKey, True)
        self.timepoint = 0
        self.directActivation = 0
        if self.group.get_phases()[self.phase].isRandom() and self.phase == 0:
            if self.is_us:
                self.variableSalience[self.phase] = self.alphaR
            else:
                self.variableSalience[self.phase] = self.alphaR
                self.csVariableSalience[self.phase] = self.alphaN

        self.wasActive = False
        self.firstPass = True
        self.isStored = False
        self.trialCount -= min(self.trialCount, currentTrials)

    @property
    def sub_E_wts(self):
        return self.subelementWeights

    def reset_completley(self):
        self.notTotal = True
        self.isA = False
        self.isB = False

        self.name = self.name
        self.names = []

        self.trialTypeCount = defaultdict(int)
        # if config.USNames.is_us(name=self.get_name()):
        #     self.names.append(self.get_name())

        for sp in self.group.get_phases():
            self.iti = max(self.iti, sp.get_ITI().get_minimum())

        self.presences = [0.0] * self.totalStimuli
        self.totalCSError = [0.0] * self.group.get_no_of_phases()
        self.totalUSError = [0.0] * self.group.get_no_of_phases()
        self.oldUSError = 0
        self.averageUSError = 0
        self.oldCSError = 0
        self.averageCSError = 0
        self.variableSalience = [0.0] * self.group.get_no_of_phases()
        self.csVariableSalience = [0.0] * self.group.get_no_of_phases()
        self.elementCurrentWeightsKey = f"{self.group.get_name_of_grp()}{self.parent.get_name()}{self.microstimulusIndex}elementCurrentWeights"
        self.group.clear_map(self.elementCurrentWeightsKey)

        self.elementPredictionKey = f"{self.group.get_name_of_grp()}{self.parent.get_name()}{self.microstimulusIndex}elementPrediction"
        self.group.clear_map(self.elementPredictionKey)
        for i in range(self.group.get_no_of_phases()):
            self.group.add_to_map(str(i), [[0.0] * self.totalMax for _ in range(
                self.totalStimuli)], self.elementCurrentWeightsKey, True)

            self.group.add_to_map(str(i), [[0.0] * self.totalMax for _ in range(
                self.totalStimuli)], self.elementPredictionKey, True)

        self.group.clear_map(map_name=self.aggregateActivationsKey)
        self.group.clear_map(map_name=self.aggregateSaliencesKey)
        self.group.clear_map(map_name=self.aggregateCSSaliencesKey)

        for i in range(self.totalTrials+1):
            self.group.add_to_map(
                str(i), [0.0] * (self.totalMax + self.iti), self.aggregateActivationsKey, True)
            self.group.add_to_map(
                str(i), [0.0] * (self.totalMax + self.iti), self.aggregateSaliencesKey, True)
            self.group.add_to_map(
                str(i), [0.0] * (self.totalMax + self.iti), self.aggregateCSSaliencesKey, True)

        self.group.clear_map(map_name=self.eligibilitiesKey)

        for i in range(self.totalStimuli):
            self.group.add_to_map(
                str(i), [0.0] * self.totalMax, self.eligibilitiesKey, True)

    def reset_activation(self):
        self.activation = 0.0
        self.directActivation = 0
        self.subelementActivations = [0] * int(self.subelementNumber)

    def set_active(self, name: str, active_status: bool, duration_point: float):
        if active_status and not self.disabled:
            self.wasActive = True
        self.durationPoint = duration_point
        # this is for cpd
        if hasattr(self, 'isA') and not self.isA:
            self.isA = (self.a is not None) and (self.a.get_has_been_active())
            # self.isA = hasattr(
            #     self, 'a') and self.a is not None and self.a.get_has_been_active()
        if hasattr(self, 'isB') and not self.isB:

            self.isB = (self.b is not None) and (self.b.get_has_been_active())
            # self.isB = hasattr(
            #     self, 'b') and self.b is not None and self.b.get_has_been_active()
        self.setParams()

    def update_activation(self, name: str, presence: float, duration: float, microstimulusIndex: int):
        self.microstimulusIndex = microstimulusIndex
        self.microPlus = microstimulusIndex+1
        # self.time =  1-presence if self.presenceMean else self.durationPoint
        self.time = self.durationPoint-1
        # print(self.time, self.name)
        if config.USNames.is_us(name=name):
            if self.time >= self.microIndex:
                self.time = max(0, self.time - self.usPersistence)

        self.difference = self.time - (self.ratio*(self.microIndex-1))
        if self.difference < 0:
            self.difference *= self.cscLikeness

        # if self.is_us and self.difference < 0:
        #     self.difference = 0

        self.numerator = math.pow(self.difference, 2)
        self.new_value = math.exp(-(self.numerator) /
                                  self.denominator) if presence > 0 else 0
        # print('new value', presence,  self.name, self.new_value)

        if math.isnan(self.numerator) or math.isinf(self.numerator) or math.isnan(self.denominator) or math.isinf(self.denominator) or math.isnan(self.new_value):
            pass
        if self.is_context:
            self.new_value *= presence
        self.maxActivation = max(self.new_value, self.maxActivation)
        if name == self.get_name():

            self.activation = self.new_value * self.intensity
            self.activation = 0 if self.disabled else self.activation

        if self.get_name() == name:
            self.directActivation = self.activation

    def getAsymptote(self) -> float:
        # DynaAsy = self.dynamic_asymptote(low_bound=0.0,upper_bound=1.1,step=0.1,v=0.9)
        # df = self.conversion(dyna_asy_mat=DynaAsy, lower_bound=0, upper_bound=11)

        # pred_index = np.round(self.getDirectActivation(),1)
        # out_index = np.round(el2.directActivation, 1)
        # #el.getAsymptote()
        # error1 = df.loc[pred_index][out_index]

        if self.timepoint <= self.parent.get_last_onset():
            # df.loc
            if self.assoc > 0.9 or self.getDirectActivation() > 0.1 * self.intensity:
                self.asy = 1
            else:
                self.asy = 0
        else:
            if self.getDirectActivation() > 0.1 * self.intensity:
                self.asy = 1
            else:
                self.asy = 0
        return self.asy

    def getAlpha(self) -> float:
        # self.beta * self.intensity if self.is_us else self.salience
        return self.salience * self.intensity if self.is_us else self.salience

    def getDirectActivation(self) -> float:
        return self.directActivation

    def getGeneralActivation(self) -> float:
        return self.generalActivation

    def getParent(self):
        return self.parent

    def storeAverageUSError(self, d: float, act: float):
        self.oldUSError = self.averageUSError
        if self.averageUSError == 0:
            self.averageUSError = max(0, self.alphaR)
        else:
            self.averageUSError = self.averageUSError * \
                (1 - act / self.factor) + d * act / self.factor

    def storeAverageCSError(self, d: float, act: float):
        self.oldCSError = self.averageCSError
        if self.averageCSError == 0:
            self.averageCSError = max(0, self.alphaN)
        else:
            self.averageCSError = self.averageCSError * \
                (1 - act / self.factor) + d * act / self.factor

    def random_with_range(self, min_val, max_val):
        return random.randint(min_val, max_val)

    def update_assoc_trace(self, assoc):
        self.asoc = max(0, min(1.0, assoc * self.vartheta))

        self.assoc = self.asoc
        # print(self.subelementNumber)

        # self.count = 0
        # self.total_activation = max(self.asoc, self.directActivation)

        # while self.count < self.subelementNumber:
        #     i = self.random_with_range(0, int(self.subelementNumber - 1))

        #     if self.subelementActivations[i] == 0:
        #         self.count += 1

        #     self.subelementActivations[i] = 1 if (self.subelementActivations[i] == 1 or
        #                                         random.random() < self.total_activation) else 0

        self.count = 0
        self.total_activation = max(self.asoc, self.directActivation)

        # while self.count < self.subelementNumber:
        #     i = self.random_with_range(0, int(self.subelementNumber - 1))

        #     if self.subelementActivations[i] == 0:
        #         self.count += 1

        #     self.subelementActivations[i] = 1 if (self.subelementActivations[i] == 1 or
        #                                         random.random() < self.total_activation) else 0

        select_ele = random.sample(population=list(
            range(self.subelementNumber)), k=random.randint(a=0, b=self.subelementNumber))
        # select_ele = random.sample(population=list(
        #     range(self.subelementNumber)), k=self.subelementNumber)
        mask = np.isin(range(self.subelementNumber), select_ele).astype(int)

        for i in range(len(mask)):
            self.subelementActivations[i] = mask[i]

        self.generalActivation = max(0, max(self.assoc, self.activation))
        # print(self.generalActivation, self.get_name())

        ############## saving the activity ####################
        # print(self.trialCount)

        self.temp = [0.0] * (self.totalMax + self.iti)
        ob3 = self.group.get_from_db(
            str(self.trialCount), self.aggregateActivationsKey)
        if ob3 is not None:
            self.temp = ob3

        self.temp[self.timepoint] = self.activation  # Saving the DA
        self.group.add_to_map(str(self.trialCount),
                              self.temp, self.aggregateActivationsKey, True)
        # print(self.name, self.temp, self.timepoint)

    @property
    def get_activation(self):
        select_ele = random.sample(population=list(
            range(self.subelementNumber)), k=random.randint(a=0, b=self.subelementNumber))
        mask = np.isin(range(self.subelementNumber), select_ele).astype(int)
        for i in range(len(mask)):
            self.subelementActivations[i] = mask[i]
        return self.subelementActivations

    def getPrediction(self, stimulus: int, element: int, current: bool, maximum: bool) -> float:
        # print(self.subelementNumber)
        # print(self.subelementWeights.shape)
        self.totalPrediction = 0.0
        # print(self.name)
        for i in range(int(self.subelementNumber)):
            # print(i)
            # * (self.directActivation > 0.1 or self.generalActivation * self.vartheta)
            self.totalPrediction += self.subelementWeights[i][stimulus][element] * (
                self.directActivation > 0.1 or self.generalActivation * self.vartheta)
        # print(halla.get())
        return self.totalPrediction

    def getCurrentUSError(self) -> float:
        return self.averageUSError

    def getCurrentCSError(self) -> float:
        return self.averageCSError

    def getVariableSalience(self) -> float:
        return self.variableSalience[self.phase]

    def getCSVariableSalience(self) -> float:
        return self.csVariableSalience[self.phase]

    def getTotalError(self, abs: float) -> float:
        self.totalUSError[self.phase] = 0.9997 * \
            self.totalUSError[self.phase] + 0.0003 * abs
        return self.totalUSError[self.phase]

    def getTotalCSError(self, abs: float) -> float:
        self.totalCSError[self.phase] = 0.99997 * \
            self.totalCSError[self.phase] + 0.00003 * abs
        return self.totalCSError[self.phase]

    def setVariableSalience(self, vs: float):
        self.temp = [0.0] * (self.totalMax + self.iti)
        ob3 = self.group.get_from_db(
            str(self.trialCount), self.aggregateSaliencesKey)
        if ob3 is not None:
            self.temp = ob3
        self.temp[self.timepoint] = (self.temp[self.timepoint] * (
            self.combination - 1) + self.variableSalience[self.phase]) / self.combination

        self.group.add_to_map(str(self.trialCount), self.temp,
                              self.aggregateSaliencesKey, True)
        self.variableSalience[self.phase] = vs if vs > 0.001 else 0

    def setCSVariableSalience(self, vs: float):
        self.temp = [0.0] * (self.totalMax + self.iti)
        ob3 = self.group.get_from_db(
            str(self.trialCount), self.aggregateCSSaliencesKey)
        if ob3 is not None:
            self.temp = ob3
        self.temp[self.timepoint] = (self.temp[self.timepoint] * (
            self.combination - 1) + self.csVariableSalience[self.phase]) / self.combination
        self.group.add_to_map(str(self.trialCount), self.temp,
                              self.aggregateCSSaliencesKey, True)
        self.csVariableSalience[self.phase] = vs if vs > 0.001 else 0

    def increment_timepoint(self, time: int) -> str:
        self.hasReset = False
        # self.timepoint+=1
        # return self.elementDAKey
        return (self.elementCurrentWeightsKey, self.elementPredictionKey)

    def isCommon(self) -> bool:
        return len(self.get_name()) > 1

    def getAssoc(self) -> float:
        return self.assoc

    def getDurationPoint(self) -> float:
        return self.durationPoint

    def updateElement(self, otherActivation, otherAlpha, other, ownError, otherError, otherName, group):

        DynaAsy = Phase.SimPhase.dynamic_asymptote(
            low_bound=0.0, upper_bound=1.1, step=0.1, v=0.9)
        df = Phase.SimPhase.conversion(
            dyna_asy_mat=DynaAsy, lower_bound=0, upper_bound=11)

        self.nE2 = abs(ownError)
        self.nE = otherError
        self.c1 = (otherName in self.names)
        self.totalWeight = 0.0
        self.totalPrediction = 0.0
        self.c1 = (otherName in self.names)
        ob1 = self.group.get_from_db(
            str(self.names.index(otherName)), self.eligibilitiesKey)
        self.temp = [0.0] * self.totalMax
        if ob1 is not None:
            self.temp = ob1
        self.val1 = (other.getAssoc(
        ) / (self.temp[other.get_microIndex()] + 0.001)) if self.c1 else 1.0
        self.c2 = (other.getAssoc() == 0 or self.val1) > 0.9
        if self.c1 and self.c2:
            self.presences[self.names.index(
                otherName)] = other.parent.get_was_active()
        if self.c1:
            self.temp[other.get_microIndex()] = max(
                self.temp[other.get_microIndex()] * 0.95, other.getAssoc())
            self.group.add_to_map(str(self.names.index(otherName)),
                                  self.temp, self.eligibilitiesKey, True)

        # self.index = self.names.index(otherName) if otherName in self.names else -1
        # # print(self.index)
        # if self.index == -1:
        #     self.names.append(otherName)
        #     self.index = self.names.index(otherName)

        self.eligi = 0
        # print('out of bounfs', self.outOfBounds)
        # print(halla.get())
        if not self.outOfBounds:
            self.eligi = math.pow((other.getAssoc() == 0 or (
                other.getAssoc() / (self.temp[other.get_microIndex()] + 0.001))), self.exponent)
        else:
            self.eligi = 0.1
        if self.eligi > 0 and self.eligi < 0.01:
            self.eligi = 0.1

        # self.ac1 = int(self.getAsymptote())
        # self.ac2 = int(other.getAsymptote())
        self.selfDereferencer = 1.0
        if otherName in self.get_name() or self.get_name() in other.get_name():
            self.selfDereferencer = 0.05 if not self.is_us else 0
        self.index = self.names.index(
            otherName) if otherName in self.names else -1
        # print('INDEX', self.index, self.name)
        if self.index == -1:
            # print('kends')
            self.names.append(otherName)
            self.index = self.names.index(otherName)
        # self.maxDurationPoint = self.durationPoint
        # self.maxDurationPoint2 = other.getDurationPoint()
        # self.fixer = 1
        # if self.directActivation > 0.1:
        #     if self.maxDurationPoint >= self.maxDurationPoint2:
        #         self.fixer = 1.0
        #     else:
        #         self.fixer = self.parent.get_b()
        # self.totalWeight = 0.0
        # self.totalPrediction = 0.0
        # self.x1 = min(1, max(self.ac1, self.assoc * 0.9))
        # self.x2 = min(1, max(self.ac2, other.getAssoc() * 0.9))
        # self.x3 = (self.fixer * self.x2 - abs(self.x1 - self.x2)) / \
        #     max(self.x1 + 0.001, self.x2 + 0.001)
        # self.nE = (otherError - self.ac2 * 1.0) + self.x3 * 1.0
        # if other.is_us:
        #     self.asymptote = self.nE

        # self.Dereferencer = 1.0
        # if otherName in self.get_name() or self.get_name() in other.get_name():
        #     self.Dereferencer = 0.01 if not self.is_us else 0

        pred_index = np.round(self.directActivation, 1)
        out_index = np.round(other.directActivation, 1)
        self.asymptote = df.loc[pred_index][out_index]
        # print('asymp', self.asymptote)

        self.currentVariableSalience = (
            other.is_us * self.variableSalience[self.phase]) if other.is_us else (self.csVariableSalience[self.phase])
        # print(self.currentVariableSalience, self.get_name(), otherName)
        # print(halla.get())
        # self.curSal = self.beta * self.intensity if self.is_us else self.salience
        self.curSal = self.salience * self.intensity if self.is_us else self.salience
        # print(self.curSal)
        # print(halka.get())

        # self.commonDiscount = (self.subelementNumber / self.totalElements) if self.isCommon()else(1.0 - self.group.get_model().getProportions().get(self.group.get_name_of_grp()).get(self.get_name())) if len(self.parent.get_common_map()) > 0 else 1.0
        # removed the doube the error
        #
        self.tempDelta = (1.0 / (self.subelementNumber * len(self.parent.get_list()))) *\
            self.selfDereferencer * other.getAlpha() * self.curSal * self.nE2 * \
            self.nE * self.currentVariableSalience * \
            other.getGeneralActivation() * self.generalActivation

        # print('Eraaa', self.tempDelta)
        self.tempDelta *= self.salience * self.intensity if self.is_us else 1.0

        self.decay = (1 - (math.sqrt(self.curSal) / 100.0)
                      ) if self.generalActivation < 0.01 else 1.0
        # print(self.decay, 'decay')

        # print('wanna see the GA here', self.generalActivation, self.assoc, self.directActivation, self.name, self.tempDelta )
        # print(self.subelementNumber)

        for i in range(int(self.subelementNumber)):
            # print(len(self.subelementWeights[i]), self.index)
            if self.index < len(self.subelementWeights[i]) and other.get_microIndex() < len(self.subelementWeights[i][self.index]):

                # self.subelementWeights[i][self.index][other.get_microIndex()] =  self.subelementWeights[i][self.index][other.get_microIndex()]*self.decay + self.tempDelta*self.subelementActivations[i]
                self.subelementWeights[i][self.index][other.get_microIndex()] = max(-2 / self.subelementNumber, min(2 / self.subelementNumber,
                                                                                                                    self.subelementWeights[i][self.index][other.get_microIndex()]*self.decay + self.tempDelta*self.subelementActivations[i]))
                # print(self.subelementWeights)
                # print(self.subelementWeights[i][self.index][other.get_microIndex()]+self.tempDelta * self.subelementActivations[i])
                # self.subelementNumber
                # self.subelementWeights[i][self.index][other.get_microIndex()] =  (1/(1+math.exp()))
                # print(self.subelementWeights[i][self.index][other.get_microIndex()])

                # +self.tempDelta * self.subelementActivations[i]
                self.totalWeight += self.subelementWeights[i][self.index][other.get_microIndex(
                )]
                self.totalPrediction += self.subelementWeights[i][self.index][other.get_microIndex(
                )] * self.directActivation

        # print('essy', self.totalWeight, self.commonDiscount)
        # if self.group.get_model().get_isExternalSave():
        #     storeLong = self.group.createDBString(self, self.currentTrialString, other.getParent(
        #     ), self.phase, self.session, self.trialTypeCount.get(self.currentTrialString), self.timepoint, True)
        #     print(storeLong)
        #     self.group.addToDB(storeLong, self.totalPrediction)
        #     storeLong = self.group.createDBString(self, self.currentTrialString, other.getParent(
        #     ), self.phase, self.session, self.trialTypeCount.get(self.currentTrialString), self.timepoint, False)
        #     self.group.addToDB(storeLong, self.totalWeight)
        ob2 = self.group.get_from_db(
            str(self.phase), self.elementCurrentWeightsKey)
        self.current = [
            [0.0] * self.totalMax for _ in range(self.totalStimuli)]
        if ob2 is not None:
            self.current = ob2
        # if self.index < len(self.current) and other.get_microIndex() < len(self.current[self.index]):
        #     print('TRUE')
        # print(self.name, self.current, self.index, other.get_microIndex(), str(self.phase))
        self.current[self.index][other.get_microIndex()] = self.totalWeight
        # self.current[self.index][other.get_microIndex()] = self.totalWeight
        self.group.add_to_map(str(self.phase), self.current,
                              self.elementCurrentWeightsKey, True)
        if self.firstPass:
            self.timepoint += 1
            self.firstPass = False
        # print("TIMEPOINT", self.timepoint)

    def resetForNextTimepoint(self):
        if not self.hasReset:
            self.maxActivation = 0
            self.hasReset = True
            self.isStored = False
            self.firstPass = True
            self.activation = 0
            self.subelementActivations = [0] * int(self.subelementNumber)

    def store(self):
        if not self.isStored:
            self.timepoint = 0
            self.directActivation = 0
            self.subelementActivations = [0] * int(self.subelementNumber)
            self.trialCount += 1
            self.isStored = True

    @staticmethod
    def dynamic_asymptote(low_bound: float = -1.1, upper_bound: float = 1.1, step: float = 0.001, v: float = 0.66):
        # build an empty matrix
        activity_range_vals = np.arange(
            start=low_bound, stop=upper_bound, step=step, dtype=float)
        dyna_Asymptote_lookup_dict = np.zeros((len(np.arange(low_bound, upper_bound, step, dtype=float)),
                                               len(np.arange(low_bound, upper_bound, step, dtype=float))))
        # print(dyna_Asymptote_lookup_dict)

        for pred_pointer in range(len(np.arange(low_bound, upper_bound, step, dtype=float))):
            for out_pointer in range(len(np.arange(low_bound, upper_bound, step, dtype=float))):
                dyna_Asymptote_lookup_dict[pred_pointer][out_pointer] = \
                    (1 - (((activity_range_vals[pred_pointer] - activity_range_vals[out_pointer]) ** 2) / 2)) \
                    * ((v * (activity_range_vals[out_pointer])) + ((1 - v) * (activity_range_vals[pred_pointer])))

        # df = pd.DataFrame(dyna_Asymptote_lookup_dict, index=[x / 10 for x in range(low_bound,upper_bound, step)], columns=[x / 10 for x in range(low_bound,upper_bound, step)])
        # print(df)

        return dyna_Asymptote_lookup_dict

    @staticmethod
    def conversion(dyna_asy_mat: np.array, lower_bound: int = 0, upper_bound: int = 11):
        dys = pd.DataFrame(dyna_asy_mat, index=[x/(upper_bound - 1) for x in range(
            lower_bound, upper_bound)], columns=[x/(upper_bound - 1) for x in range(lower_bound, upper_bound)])
        # dys = pd.DataFrame(dyna_asy_mat)
        # print(dys)
        return dys
