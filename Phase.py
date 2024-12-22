import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import time
import config
import Trrial
import numpy as np
import random
import pandas as pd


class SimPhase:
    def __init__(self, phaseNum: int, sequence_of_stim: str, orderr: List[Any], stimuli2: Dict[str, Any], grp: Any, random: bool, timing: Any, iti_obj: Any, ctx: Any):

        self.phaseNum = phaseNum
        self.initilaSeq = sequence_of_stim
        self.orderedSeq = orderr
        self.stimuli = stimuli2
        self.group = grp
        self.random = random
        self.trials = len(self.orderedSeq)
        self.timingConfig = timing
        self.itis = iti_obj
        self.contextCfg = ctx
        self.contextCfgs = defaultdict(lambda: None)
        self.contextCfgs[ctx.get_symbol()] = ctx
        self.cues = defaultdict(lambda: None)
        self.results = defaultdict(lambda: None)
        self.all_sequence = []
        self.sessions = 1
        # self.vartheta = None

        self.tempMap = defaultdict(lambda: None)
        self.allMap = defaultdict(lambda: None)
        self.usIndexes = defaultdict(lambda: None)
        self.csIndexes = defaultdict(lambda: None)
        self.nameIndexes = defaultdict(lambda: None)
        # self.activeCS = []
        # self.trialAppearances = defaultdict(lambda: None)

        self.presentCS = set()
        self.trialLengths = []
        self.completeLengths = []
        self.presentStimuli = []
        self.activeCS = []

        self.onsetMap = defaultdict(lambda: None)
        self.offsetMap = defaultdict(lambda: None)
        self.generalOnsetMap = defaultdict(lambda: None)
        self.generalOffsetMap = defaultdict(lambda: None)
        self.csMap = defaultdict(lambda: None)

        self.usPredictions = defaultdict(lambda: None)
        self.csPredictions = defaultdict(lambda: None)
        self.averageCSErrors = defaultdict(lambda: None)

        for k, v in self.group.get_cue_map().items():
            if k in sequence_of_stim:
                self.cues[k] = v

        # print('check context here', self.cues)

    def get_cues(self):
        return self.cues

    def reset(self) -> None:
        self.results = {}
        self.cues = {}

    def run(self):
        self.results = self.cues  # new assignmentn
        # print('RESULTS SAME AS CUES', self.results)
        context = self.group.get_model().is_use_context()  # Boolean
        # print('context state', context)
        combinations = 1  # self.isRandom() if self.group.get_model().getCombinationNo() else 1
        temp_results = defaultdict(lambda: None)

        # print(self.initilaSeq)
        # print(halla.get())

        # Set some standard parameters from the model of all stim elements

        for stim in self.group.get_cue_map().values():
            # print(stim.get_name(), len(stim.get_list()))
            stim.set_reset_context(reset=self.reset_context)
            stim.post_phase_init()

            # Think about eliminating stds and persitance and taking them to element - microstim() fxn
            for s_ele in stim.get_list():
                s_ele.set_CSCV(cscv=self.std)
                s_ele.set_USCV(uscv=self.us_cv)
                s_ele.set_USScalar(
                    usScalar=self.group.get_model().get_us_scalar())
                s_ele.set_CSScalar(
                    csScalar=self.group.get_model().get_cs_scalar())
                s_ele.set_USPersistence(p=self.us_persistance)

        for i in range(1):
            temp_seq = self.orderedSeq
            # print('hey',temp_seq)
            if self.isRandom():
                for x in range(self.trials):
                    nr = random.randint(0, len(self.orderedSeq) - 1)
                    swap = temp_seq[x]
                    temp_seq.remove(swap)
                    temp_seq.insert(nr, swap)
                if i == 0:
                    for ran in range(10):
                        for x in range(self.trials):
                            nr = random.randint(0, len(self.orderedSeq) - 1)
                            swap = temp_seq[x]
                            temp_seq.remove(swap)
                            temp_seq.insert(nr, swap)


            # Swaping/ randomizztion to be done but not now
            trialIndexMap2 = defaultdict(lambda: defaultdict(lambda: None))
            trialTypeCounters2 = defaultdict(lambda: 0)

            for j in range(len(temp_seq)):
                str = temp_seq[j].get_trial_string()[1:]
                # print('yooo', str)
                trialTypeCounters2[str] = 0
                trialIndexMap2[str] = defaultdict(lambda: None)
            for j in range(len(temp_seq)):
                str = temp_seq[j].get_trial_string()[1:]
                counter = trialTypeCounters2[str]
                trialTypeCounters2[str] = counter + 1
                trialIndexMap2[str][counter] = j

            # print(trialIndexMap2)
            # print(trialTypeCounters2)

            tempRes = self.cues
            tempProbeRes = defaultdict(lambda: 0)

            for stim in self.group.get_cue_map().values():
                # Set the phase for stim and each element
                stim.set_phase(self.phaseNum-1)
                if i > 0:
                    # this means if we have more than one round of run we increment the combination, kind-of resetting everything
                    stim.incrementCombination()
                if self.isRandom():
                    stim.reset(i + 1 == combinations, 0 if i ==
                               0 else self.trials * self.sessions)

            self.algorithim(temp_seq, tempRes, context)

    def algorithim(self, sequence, tempres, context):
        # print(sequence, tempres)
        print('haloo keita begining of computation algorithm',
              self.group.get_cue_map().keys())
        self.inThisRun = set()
        self.activeList = []
        self.csActiveThisTrial = set()
        self.probeCSActiveThisTrial = set()
        wights = {}

        most_elements = 0

        for stim in self.group.get_cue_map().values():

            most_elements = max(most_elements, len(stim.get_list()))
            # print(most_elements, self.contextReset)
            stim.set_reset_context(reset=self.contextReset)  # chunguza

            for se in stim.get_list():
                if stim.is_us or se.is_us:
                    se.setIntensity(f=self.intensity)

        # Initilalize the compound cue maps that we have
        if True:
            for compound in self.group.get_cue_map().values():
                if len(compound.get_name()) > 1:
                    a = self.group.get_cue_map().get(compound.get_name()[1])
                    b = self.group.get_cue_map().get(compound.get_name()[2])
                    # Setting the stimulus to both a and b  and intiliailzaing parameters
                    compound.initialize(a, b)
                    # print('copmp', self.presentStimuli)

        # Check the status
        # for stim in self.group.get_cue_map().values():
        #     print(stim.get_name())
        #     for se in stim.cues:
        #         print(hasattr(se, 'a'))
        #         print(hasattr(se, 'b'))

        # build a container for the activations
        self.stim_pred = np.zeros((self.trials, len(self.group.get_cue_map())))
        # [
        #     [0]*self.trials for _ in range(len(self.group.get_cue_map()))]
        print(self.stim_pred)
        self.elementPredictions = [
            [0] * most_elements for _ in range(len(self.group.get_cue_map()))]

        if self.group.get_model().get_isErrors():
            self.timePointElementErrors = [[[0]*most_elements for _ in range(len(self.group.get_cue_map()))]
                                           for _ in range(self.trials * self.sessions)]
        if self.group.get_model().get_isErrors2():
            self.lastTrialElementErrors = [[[0] * most_elements for _ in range(len(self.group.get_cue_map()))]
                                           for _ in range(max(stim.get_all_maxduration()
                                                              for stim in self.group.get_cue_map().values()) * len(self.group.get_cue_map()))]
        # print('value of trials', self.trials)
        # print('stimulus dict', self.stimuli)
        # print('results dict', tempres)
        for i in range(1, self.trials*self.sessions+1):
            # print('Trial ====>', i)
            if i % self.trials == 1 and i != 1:
                self.itis.reset()  # find out what this means

                for stim in self.group.get_cue_map().values():
                    if stim.is_context:
                        stim.reset_activation(True)
            self.csActiveThisTrial.clear()
            # clear contaiers at the start of the trial you'll do later
            # count time
            # timestamp in milliseconds and stores it as an integer
            count = int(round(time.time() * 1000))
            self.currentSeq = str(sequence[i-1 % self.trials])
            # print('current Sequence', self.currentSeq)

            for stim in self.group.get_cue_map().values():
                # print(stim.get_name())
                for ele in stim.get_list():
                    ele.setCurrentTrialString(currentString=self.currentSeq)

            self.curNameSt = str(sequence[(i-1) % self.trials])
            # print('yoo', self.curNameSt)

            # Simulted stimuli ebedded in stimuli sumulation
            currentSt = self.stimuli.get(self.curNameSt)
            # print('embedded in stim sim', currentSt)

            if currentSt.isReinforced():
                self.reinforced = True
            else:
                self.reinforced = False
            # print('is reinforeced', self.reinforced)
            self.tempMap.clear()
            self.allMap.clear()
            # Extract trials recall seq was the trial arranged and [index it]
            trial = sequence[(i - 1) % self.trials].copy()  # should only be CS
            # print(trial)
            for cs in trial.get_cues():
                # From the structure of the trial  get the CS cues and map them to the temp_map
                # with the  stimulus from the temp map
                # print('CSSSS', cs)
                self.tempMap[cs] = tempres.get(cs.get_name())
            # print('temp map', self.tempMap)

            for j in range(len(sequence)):
                aTrial = sequence[j].copy()  # YOU CAN COPY COZ IT A TRIAL
                for cs in aTrial.get_cues():
                    if cs not in self.allMap:
                        self.allMap[cs] = tempres.get(cs.get_name())
            # print('--->', self.allMap)

            iti = int(round(self.itis.get_minimum()))
            timings = self.timingConfig.make_timings(self.tempMap.keys())
            # print('--> phase', timings)
            trial_length = self.timingConfig.t_max()
            # print('trials_len ---', trial_length)  # Change how you get the trial length
            us_timings = self.timingConfig.US_timings()
            # print('ustimings', us_timings)
            usOnset = 0 if us_timings.get(
                self.currentSeq[-1], -1) == -1 else us_timings.get(self.currentSeq[-1])[0]
            # print('usonsettttt', usOnset)
            usOffset = 0 if us_timings.get(
                self.currentSeq[-1], -1) == -1 else us_timings.get(self.currentSeq[-1])[1]
            # print('USoffffset', usOffset)

            self.trialLengths.append(trial_length)
            # self.completeLengths.append(trial_length+self.itis)

            self.usIndexes.clear()  # position indicies
            self.csIndexes.clear()  # position indicies
            self.usIndexCount = 0
            numberCommon = 0

            for stim in self.group.get_cue_map().values():
                # print(self.group.get_first_occurrence(s=stim), stim.get_name())
                # print(stim.get_name())
                # print('wink', self.group.get_first_occurrence(
                #     s=stim), self.get_phase_num())
                # print(self.group.get_first_occurrence(
                # s=stim) >= 0 and self.group.get_first_occurrence(s=stim) < self.get_phase_num())
                if self.group.get_first_occurrence(s=stim) >= 0 and self.group.get_first_occurrence(s=stim) < self.get_phase_num():
                    if stim.is_us:
                        self.usIndexes[stim.get_name()] = self.usIndexCount
                    else:
                        self.csIndexes[stim.get_name()] = self.usIndexCount
                        if stim.is_common():
                            numberCommon += 1
                self.usIndexCount += 1
                names = stim.get_name()
                # print('NAMES', names)
                css = [None]*len(names)
                onset = -1
                offset = trial_length
                counter = 0

                for charac in names:
                    if charac == "c":
                        pass
                    else:
                        for cs in self.tempMap.keys():

                            if cs.get_name() == charac:
                                css[counter] = cs.get_name()
                        # print(css)
                        temp_onset = timings.get(
                            css[counter], -1)[0] if css[counter] and css[counter] in timings else -1
                        onset = max(temp_onset, onset)
                        # print(onset)

                        temp_offset = timings.get(
                            css[counter], -1)[1] if css[counter] and css[counter] in timings else -1
                        offset = min(temp_offset, offset)
                        # print(offset)
                        counter += 1

                generalOnset = -1
                generalOffset = -1

                self.csMap[stim.get_name()] = css
                self.onsetMap[stim.get_name()] = onset
                self.offsetMap[stim.get_name()] = offset
                self.generalOnsetMap[stim.get_name()] = generalOnset
                self.generalOffsetMap[stim.get_name()] = generalOffset
            # Set the times of the compound well
            for stim in self.group.get_cue_map().values():
                if 'c' in stim.get_name():
                    name1 = stim.get_name()[1]
                    name2 = stim.get_name()[2]
                    # print(name1)
                    # print(name2)

                    onset = max(0, min(self.onsetMap.get(name1, 0),
                                self.onsetMap.get(name2, 0)))
                    offset = max(self.offsetMap.get(name1, 0),
                                 self.offsetMap.get(name2, 0))
                    self.onsetMap[stim.get_name()] = onset
                    self.offsetMap[stim.get_name()] = offset

            container = [[[0.0]*(trial_length+iti)
                         for _ in range(len(self.group.get_cue_map().values()))] for _ in range(len(self.group.get_cue_map().values()))]
            # print(container)
            for jj in range(1, (trial_length+iti)+1):
                print("=========>", jj)
                self.activeList.clear()
                self.activeCS.clear()
                # Set up the activations for ac stimulus at each duration point
                # print(halla.get())
                for stim in self.group.get_cue_map().values():
                    # print(stim.get_name())

                    names = stim.get_name()
                    # print("yo", names, "dnjds", self.currentSeq)
                    css = self.csMap.get(stim.get_name())
                    onset = self.onsetMap.get(stim.get_name())
                    # print('on', onset)
                    offset = self.offsetMap.get(stim.get_name())
                    # print('off', offset)
                    generalOnset = self.generalOnsetMap.get(stim.get_name())
                    # print(self.generalOnsetMap)
                    generalOffset = self.generalOffsetMap.get(stim.get_name())
                    csName = stim.get_name()
                    # print(csName)
                    stim.set_trial_length(trial_length=(trial_length+iti))
                    active = (jj >= onset and jj <= offset)
                    # print('active status', active)
                    # usActive = (jj >= us_timings.get(
                    #     self.currentSeq[-1])[0] and jj <= us_timings.get(self.currentSeq[-1])[1])
                    usActive = (jj >= usOnset and jj <= usOffset)
                    # print('usactive status', usActive)

                    if stim.is_us:
                        if stim.get_name() == '+':
                            # print('yo111')
                            # stim.set_duration(onset=usOnset, offset=usOffset, duration_point=jj - us_timings.get(
                            #     self.currentSeq[-1])[0], active=usActive & currentSt.isReinforced(), realtime=jj)

                            stim.set_duration(onset=usOnset, offset=usOffset, duration_point=jj -
                                              usOnset, active=usActive & currentSt.isReinforced(), realtime=jj)

                        else:
                            # print('yoo 2')
                            # print('which is', stim.get_name())
                            # stim.set_duration(onset=usOnset, offset=usOffset, duration_point=jj - us_timings.get(
                            #     self.currentSeq[-1])[0], active=usActive & currentSt.isReinforced(), realtime=jj)
                            stim.set_duration(onset=usOnset, offset=usOffset, duration_point=jj -
                                              usOnset, active=usActive & currentSt.isReinforced(), realtime=jj)
                        generalOnset = usOnset
                        generalOffset = usOffset
                        # stim.increment_timepoint(jj, jj > trial_length)

                    elif self.contextCfg.get_context() == csName:
                        stim.set_duration(
                            onset=0, offset=trial_length+iti-1, duration_point=jj, active=True, realtime=jj)
                        generalOnset = 0
                        generalOffset = trial_length+iti-1
                        # stim.increment_timepoint(jj, jj > trial_length)

                    elif stim.is_cs:
                        # print(stim.get_symbol())
                        stim.set_duration(
                            onset=onset, offset=offset, duration_point=jj-onset, active=active, realtime=jj)
                        generalOnset = onset
                        generalOffset = offset
                        # stim.increment_timepoint(jj, jj > trial_length)

                    elif self.contextCfg.get_context() != csName and stim.is_context:
                        stim.set_duration(
                            onset=0, offset=trial_length+iti-1, duration_point=jj, active=False, realtime=jj)

                        generalOnset = 0
                        generalOffset = trial_length+iti-1
                        # stim.increment_timepoint(jj, jj > trial_length)

                self.usPredictions.clear()
                self.csPredictions.clear()

                if self.elementPredictions is not None:
                    for us_keys in self.usIndexes.keys():
                        current_us = self.group.get_cue_map()[
                            us_keys]  # Stimulus itself
                        tempPrediction = 0.0
                        div_no = len(current_us.get_list())
                        # print(div_no)

                        # Stim.get return cues [index] cues for stim are list so it retrives the element of us e0
                        for ele_idx in range(len(current_us.get_list())):
                            tempPrediction += abs((current_us.get(index=ele_idx).getAsymptote(
                            ) - self.elementPredictions[self.usIndexes.get(us_keys)][ele_idx])/div_no)

                        # self.usPredictions[us_keys] = abs(tempPrediction)
                        if abs(tempPrediction) > 0.05:
                            self.usPredictions[us_keys] = tempPrediction

                    for cs_keys in self.csIndexes.keys():
                        # print('CS keys', cs_keys)
                        current_cs = self.group.get_cue_map().get(cs_keys)
                        tempPrediction = 0.0
                        div = len(current_cs.get_list())

                        for k2 in range(len(current_cs.get_list())):
                            tempPrediction += abs((current_cs.get(index=k2).getAsymptote(
                            ) - self.elementPredictions[self.csIndexes.get(cs_keys)][k2]) / div)

                        # self.csPredictions[cs_keys] = abs(tempPrediction)

                        if abs(tempPrediction) > 0.05:
                            self.csPredictions[cs_keys] = tempPrediction
                        # else:
                        #     self.csPredictions[cs_keys] = 0

                averageError = 0.0
                for s in self.usPredictions.keys():
                    averageError += self.usPredictions.get(
                        s)/len(self.usPredictions)

                for s in self.group.get_cue_map().keys():  # 'Ω', 'A', 'B', '+', 'cAB']
                    tempError = 0.0

                    factor = 0 if self.group.get_cue_map().get(s).is_us else 1
                    correction = 1 if self.group.get_cue_map().get(
                        s).is_context or self.group.get_cue_map().get(s).is_us else 0

                    for s2 in self.csPredictions.keys():  # 'Ω', 'A', 'cAB')
                        if s != s2 and s not in s2 and s2 not in s:
                            # print('smile', s, s2)
                            tempError += self.csPredictions.get(s2)/max(
                                1, len(self.csPredictions) - factor - correction)

                    self.averageCSErrors[s] = tempError
                # print('interested in this', self.averageCSErrors)
                shouldUpdateUS = any(self.group.get_cue_map().get(
                    naming).get_should_update() for naming in self.usIndexes.keys())
                # print('shouldus', shouldUpdateUS)
                shouldUpdateCS = any(self.group.get_cue_map().get(
                    naming).get_should_update() for naming in self.csIndexes.keys())

                stimCount1 = 0

                for cue in self.group.get_cue_map().values():
                    csName = cue.get_name()

                    for el in cue.get_list():

                        el.update_assoc_trace(self.elementPredictions[stimCount1][el.get_microIndex(
                        )] if self.elementPredictions is not None else 0.0)

                    stimCount1 += 1

                stimCount = 0
                for cue in self.group.get_cue_map().values():
                    csName = cue.get_name()
                    # print(csName)
                    counterBBB = 0

                    for el in cue.get_list():

                        act = el.getGeneralActivation() * el.getParent().get_salience()
                        rightTime = j % max(1, usOnset) <= (usOffset - usOnset)
                        rightTime = False
                        if shouldUpdateUS or True or len(self.usIndexes) == 0 or (cue.is_context and rightTime):
                            if cue.is_us:  # My addtion
                                el.storeAverageUSError(abs(averageError), act)
                        if shouldUpdateCS or True or len(self.csIndexes) == 0:
                            if not cue.is_us:
                                el.storeAverageCSError(
                                    abs(self.averageCSErrors.get(cue.get_name())), act)

                        sumOfPredictions = 0.0
                        for cue2 in self.group.get_cue_map().values():
                            currentElementPrediction = 0.0
                            for el2 in cue2.get_list():
                                prediction = el2.getPrediction(el2.get_names().index(
                                    el.get_name()), el.get_microIndex(), True, False)
                                currentElementPrediction += prediction
                            sumOfPredictions += currentElementPrediction

                        self.elementPredictions[stimCount][counterBBB] = sumOfPredictions
                        counterBBB += 1

                    ob3 = self.group.get_from_db(
                        str(self.phaseNum), el.elementPredictionKey)
                    self.current2 = [
                        [0.0] * most_elements for _ in range(self.group.get_total_stimuli())]
                    # print(self.current2)
                    if ob3 is not None:
                        self.current2 = ob3
                        # print('yooo', self.current2)
                    # if self.index < len(self.current) and other.get_microIndex() < len(self.current[self.index]):
                    #     print('TRUE')
                    # print(self.name, self.current, self.index, other.get_microIndex(), str(self.phase))
                    self.current2 = self.elementPredictions
                    # print(self.current2)
                    # self.current[self.index][other.get_microIndex()] = self.totalWeight
                    self.group.add_to_map(str(self.phaseNum), self.current2,
                                          el.elementPredictionKey, True)
                    # print(self.elementPredictions)
                    # print(hlaa.get())
                    print(np.average(
                        self.elementPredictions, axis=1))
                    print('aye')
                    self.stim_pred[(i-1)] = np.average(
                        self.elementPredictions, axis=1)
                    self.csActiveThisTrial.add(csName)
                    self.activeList.append(cue)
                    self.inThisRun.add(csName)
                    stimCount += 1

                # print(self.stim_pred)

                counter1 = 0
                for stim in self.group.get_cue_map().values():
                    # print('stim_name', stim.get_name())
                    if stim.get_name() not in self.nameIndexes:
                        self.nameIndexes[stim.get_name()] = counter1

                    for el in stim.get_list():
                        # timings.get(CS.US)[1]
                        # if jj < us_timings.get(self.currentSeq[-1])[1] and self.group.get_model().isErrors:
                        #     self.timePointElementErrors[i - 1][counter1][el.get_microIndex()] = (el.getAsymptote(
                        #     ) - self.elementPredictions[counter1][el.get_microIndex()]) / us_timings.get(self.currentSeq[-1])[1]
                        if jj < usOffset and self.group.get_model().isErrors:
                            self.timePointElementErrors[i - 1][counter1][el.get_microIndex()] = (el.getAsymptote(
                            ) - self.elementPredictions[counter1][el.get_microIndex()]) / usOffset

                        if i == self.trials * self.sessions and self.group.get_model().isErrors2:
                            self.lastTrialElementErrors[jj - 1][counter1][el.get_microIndex()] = (
                                el.getAsymptote() - self.elementPredictions[counter1][el.get_microIndex()])

                        averageError = el.getCurrentUSError()
                        # print('yoo', averageError)
                        averageCSError = el.getCurrentCSError()
                        tempDopamine = el.getVariableSalience()
                        tempCSDopamine = el.getCSVariableSalience()

                        act = el.getGeneralActivation() * el.getParent().get_salience()
                        # print(act)
                        threshold = 0.3 if stim.is_context else 0.4
                        threshold2 = 0.9 if stim.is_context else 0.9

                        rightTime = j % max(1, usOnset) <= (usOffset - usOnset)
                        rightTime = False
                        if shouldUpdateUS or True or len(self.usIndexes) == 0 or (stim.isContext and rightTime):
                            totalErrorUS = el.getTotalError(abs(averageError))
                            # tempDopamine = tempDopamine * (1 - self.integration * act) * (1 - act * (totalErrorUS > threshold ? totalErrorUS / 100.0 : 0)) + (self.integration * act * max(0, min(1, abs(averageError))) * max(el.getParent().getWasActive(), el.getGeneralActivation()))
                            tempDopamine = tempDopamine * (1 - self.integration * act) * (
                                1 - act * (totalErrorUS / 100.0 if totalErrorUS > threshold else 0))
                            + (self.integration * act * np.clip(abs(averageError), 0, 1) *
                               max(el.getParent().get_was_active(), el.getGeneralActivation()))

                        if shouldUpdateCS or True or len(self.csIndexes) == 0:
                            totalErrorCS = el.getTotalCSError(
                                abs(averageCSError))
                            # tempCSDopamine = tempCSDopamine * (1 - self.csIntegration * act) * (1 - (totalErrorCS > threshold2 ? totalErrorCS / 100.0 : 0)) + self.csIntegration * act * max(0, min(1, abs(averageCSError)) * max(el.getParent().getWasActive(), el.getGeneralActivation()))
                            # temp_cs_dopamine = temp_cs_dopamine * (1 - self.cs_integration * act) * (1 - (totalErrorCS / 100.0 if totalErrorCS > threshold2 else 0)) + self.cs_integration * act * max(0, min(1, abs(average_cs_error)) * max(el.get_parent().get_was_active(), el.get_general_activation()))
                            tempCSDopamine = tempCSDopamine * (1 - self.cs_integration * act) * (1 - (totalErrorCS / 100.0 if totalErrorCS > threshold2 else 0)) + \
                                self.cs_integration * act * np.clip(abs(averageCSError), 0, 1) * max(
                                    el.getParent().get_was_active(), el.getGeneralActivation())
                        if stim.is_us:
                            el.setVariableSalience(tempDopamine)
                        el.setCSVariableSalience(tempCSDopamine)

                    counter1 += 1

                # print(self.nameIndexes)

                for stim in self.group.get_cue_map().values():
                    # print('names of stimulus we go through in the increment timepoimt', stim.get_name())
                    stim.increment_timepoint(jj, jj > trial_length)

                self.update_cues(0, tempres, self.tempMap.keys(), jj)

                for cl in self.group.get_cue_map().values():
                    cl.reset_for_next_timepoint()

            #     for s in self.group.get_cue_map().values():
            #         wights.setdefault(s.get_name(), [])

            #         wights[s.get_name()].append(s.get_trial_average_weights_a(self.group.get_trial_type_index(s = self.currentSeq[1:]), self.get_phase_num() - 1))

            # self.group.compactDB()
            # print(halla.get())
            self.store(self.group.get_cue_map(),
                       self.csActiveThisTrial, self.currentSeq)

            # [s.get_trial_average_weights(0, 0))

            # print('sndjsnjs', halla.set())

            # self.activeLastStep.clear()
            # self.csActiveLastStep.clear()
            # self.control.setEstimatedCycleTime(time.time() * 1000 - count)

    def store(self, temp_res: Dict[str, Any], current: Set[str], current_sequence: str) -> None:
        for cue in self.group.get_cue_map().values():
            cue.prestore()
        for cue in self.group.get_cue_map().values():
            cue.store(current_sequence)
        for cue in self.group.get_cue_map().values():
            cue.post_store()

# MODEL MUST HAVES FROM INITILIZATIONS - paramters set from the modlel from external paramz

    def set_gamma(self, gma):
        self.gamma = gma

    def set_cs_scalar(self, value: float):
        self.cs_scalar = value

    def set_vartheta(self, v: float):
        self.vartheta = v

    def set_parameters(self):
        if self.group.get_cue_map() is not None and self.std is not None:
            for st in self.group.get_cue_map().values():
                st.set_parameters(
                    st.is_us * self.us_cv if st.is_us else self.std, self.vartheta)

    def set_leak(self, us: float, cs: float) -> None:
        leak = 1 - us
        self.integration = us
        cs_leak = 1 - cs
        self.cs_integration = cs

    def set_delta(self, delta):
        self.delta = delta

    def set_self_pred(self, d: float):
        self.discount = d

    def set_context_salience(self, salience: float) -> None:
        self._salience = salience

    def set_reset_context(self, r: bool):
        self.reset_context = r

    def set_std(self, s: float):
        self.std = s

    def set_us_std(self, s: float):
        self.us_cv = s

    def set_us_persistance(self, p: float):
        self.us_persistance = p

    def set_cs_c_like(self, b: float):
        for stim in self.group.get_cue_map().values():
            stim.set_csc_like(b)

    def set_val_context_reset(self, v: float):
        self.contextReset = v

    def set_subset_size(self, ii):
        self.subsetsize = ii

    def set_intensity(self, i):
        self.intensity = i

    def get_gamma(self):
        return self.gamma

    def get_delta(self):
        return self.delta

    def get_ITI(self) -> Any:
        return self.itis

    def get_context_config(self) -> Dict[str, Any]:
        return config.ContextConfig()

    def get_intensity(self):
        return self.intensity

    def getPresentCS(self) -> Set[Any]:
        return self.presentCS

    def get_current_iti(self):
        return self.itis.get_minimum()

    def set_present_cs(self, present_cs: Set[Any]) -> None:
        self.presentCS = present_cs

    def isRandom(self):
        return self.random

    def containsStimulus(self, s: Any) -> bool:
        # print('yoooo', self.presentStimuli)
        return s in self.presentStimuli

    def get_phase_num(self) -> int:
        return self.phaseNum

    # def add_context_config(context_cfg: Any) -> None:
    #     context_cfgs[context_cfg.get_symbol()] = context_cfg

    def create_subsets(self):
        self.subsets = [[] for _ in self.group.get_total_max]
        for stim in self.group.get_cue_map().values():
            if stim.is_cs:
                for elemz in stim.get_list():
                    if len(self.subsets) > 0:
                        self.subsets[elemz.get_microIndex()].append(elemz)

    def get_subsets(self):
        return self.subsets

    def get_current_trial(self) -> str:
        return self.currentSeq[1:]

    def update_cues(self, beta_error: float, temp_res: Dict[str, Any], set_cs: Set[Any], time: int) -> None:
        DynaAsy = self.dynamic_asymptote(
            low_bound=0.0, upper_bound=1.1, step=0.1, v=0.9)
        df = self.conversion(dyna_asy_mat=DynaAsy,
                             lower_bound=0, upper_bound=11)
        # print(df)
        first_count = 0
        for cue in self.group.get_cue_map().values():
            # print('ctx FIXXXX', cue.get_name(), cue.ctx_fix )
            count2 = 0
            for cue2 in self.group.get_cue_map().values():
                if cue2.get_name() not in cue.get_names():
                    print('FALSE')
                    cue.get_names().append(cue2.get_name())
                for el in cue.get_list():
                    for el2 in cue2.get_list():
                        # print(cue.get_name(), cue2.get_name())
                        # if el2.get_name() in el.names:
                        #     pass
                        # else:
                        #     el.names.append(el2.get_name())

                        # print('pred===>', el.getAssoc(), el.name, el.names)
                        # print('out====>', el2.getAssoc(), el2.name, el2.names)

                        pred_index = np.round(el.directActivation, 1)
                        out_index = np.round(el2.directActivation, 1)
                        # print(cue.get_name(), pred_index, out_index, cue2.get_name())
                        #
                        # el.getAsymptote()
                        error1 = df.loc[pred_index][out_index] - \
                            self.elementPredictions[first_count][el.get_microIndex(
                            )]
                        # print('e1', error1)
                        
                        # 
                        error2 = df.loc[pred_index][out_index]- \
                            self.elementPredictions[count2][el2.get_microIndex(
                            )]
                        # print('e2', error2)
                        el.updateElement(el2.getDirectActivation(), el2.getAlpha(
                        ), el2, error1, error2, cue2.get_name(), self.group)
                count2 += 1
            first_count += 1

    @ staticmethod
    def dynamic_asymptote(low_bound: float = -1.1, upper_bound: float = 1.1, step: float = 0.001, v: float = 0.66):
        # build an empty matrix
        activity_range_vals = np.arange(
            start=low_bound, stop=upper_bound, step=step, dtype=float)
        dyna_Asymptote_lookup_dict = np.zeros((len(np.arange(low_bound, upper_bound, step, dtype=float)),
                                               len(np.arange(low_bound, upper_bound, step, dtype=float))))
        # print(dyna_Asymptote_lookup_dict)

        for pred_pointer in range(len(np.arange(low_bound, upper_bound, step, dtype=float))):
            for out_pointer in range(len(np.arange(low_bound, upper_bound, step, dtype=float))):
                dyna_Asymptote_lookup_dict[pred_pointer][out_pointer] = (1 - (((activity_range_vals[pred_pointer] - activity_range_vals[out_pointer]) ** 2) / 2)) \
                    * ((v * (activity_range_vals[out_pointer])) + ((1 - v) * (activity_range_vals[pred_pointer])))

        # df = pd.DataFrame(dyna_Asymptote_lookup_dict, index=[x / 10 for x in range(low_bound,upper_bound, step)], columns=[x / 10 for x in range(low_bound,upper_bound, step)])
        # print(df)

        return dyna_Asymptote_lookup_dict

    @ staticmethod
    def conversion(dyna_asy_mat: np.array, lower_bound: int = 0, upper_bound: int = 11):
        dys = pd.DataFrame(dyna_asy_mat, index=[x/(upper_bound - 1) for x in range(
            lower_bound, upper_bound)], columns=[x/(upper_bound - 1) for x in range(lower_bound, upper_bound)])
        # dys = pd.DataFrame(dyna_asy_mat)
        # print(dys)
        return dys

    @ property
    def get_predictions(self):
        return self.stim_pred
