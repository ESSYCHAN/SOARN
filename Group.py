from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import config
import re
import math

import stimm
import Trrial
import Sim_Stim
import iti
import Phase


class Grp:
    def __init__(self, name: str, num_phases: int, model: Any, **kwargs) -> None:
        self.name_of_grp = name
        self.no_of_phases = num_phases
        # self.phases = [None]*num_phases
        self.phases = []
        self.phase_trials = [0] * num_phases

        # is the maximum duration of a trial - minus the iti will be assigned later
        self.total_max = 0
        self.raw_stim = {}  # Configurations not the raw data

        for key, values in kwargs.items():
            self.raw_stim[values.get_symbol()] = values
            # if values.is_cs:
            #     self.raw_stim[str(values.symbol)] = values
            # elif values.is_us:
            #     self.raw_stim[str(values.symbol)] = values
            # elif values.is_context:
            #     self.raw_stim[str(values.symbol)] = values
            # else:
            #     raise ValueError

        # print('raw_stim configs', self.raw_stim)
        self.maps = defaultdict(dict)
        self.cache = defaultdict(dict)
        self.cues = defaultdict(dict)
        self.commons = defaultdict(dict)

        self.trial_strings = []
        self.trial_strings2 = []
        self.appearance_list = defaultdict(int)
        self.set_model(model)

    def set_model(self, model: Any):
        # function needed to initilate the model in the Grp structure
        self.model = model

    def get_total_max(self):
        return self.max_duration

    def get_name_of_grp(self):
        return self.name_of_grp

    def get_maps(self) -> Dict[str, Any]:
        return self.maps

    def add_to_map(self, key: str, entry: Any, map_name: str, add_now: bool):
        if not add_now:
            self.cache[map_name][key] = entry
            if len(self.cache[map_name]) > 100:
                for s in self.cache[map_name].keys():
                    self.maps[map_name][s] = self.cache[map_name][s]
                self.cache[map_name].clear()
        else:
            self.maps[map_name][key] = entry

    def get_from_db(self, key: str, mapname):
        return self.maps[mapname].get(key, None)

    def make_map(self, name):
        temp_disk = defaultdict(dict)
        # temp_disk[name] = None
        self.cache[name] = defaultdict(dict)
        self.maps[name] = temp_disk
        return temp_disk

    def clear_map(self, map_name: str):
        self.maps[map_name].clear()

    def createDBString(self, se: Any, trial_string: str, other: Any, phase: int, session: int, trial: int, timepoint: int, is_a: bool) -> int:
        if not self.hash_set:
            self.initialize_db_keys()
            self.hash_set = True

        key = (((((((((is_a and 1 or 0) << self.bytes5 + ((session - 1) * self.trial_types + self.total_key_set.index(trial_string))) << self.bytes4) + timepoint) << self.bytes3) + trial) << self.bytes2) +
               se.get_micro_index()) << self.bytes1) + (self.total_stimuli * self.total_stimuli * phase + self.total_stimuli * self.get_names().index(se.get_name()) + self.get_names().index(other.get_name()))
        return key

    def initialize_db_keys(self):
        self.max_sessions = 1
        self.total_key_set = []
        for i in range(self.no_of_phases):
            self.max_sessions = max(
                self.max_sessions, self.model.get_group_session(self.name_of_group, i + 1))
            for j in range(len(self.get_phases()[i].get_ordered_seq())):
                s = self.get_phases()[i].get_ordered_seq()[j].to_string()
                if s not in self.total_key_set:
                    self.total_key_set.append(s)

        self.trial_types = len(self.total_key_set)

        self.bytes1 = self.get_c(
            self.total_stimuli * self.total_stimuli * self.no_of_phases)
        self.s1 = self.bytes1
        self.bytes2 = self.get_c(self.max_micros)
        self.s2 = self.bytes1 + self.bytes2 + 1
        self.bytes3 = self.get_c(self.total_trials)
        self.s3 = self.bytes1 + self.bytes2 + self.bytes3 + 2
        self.bytes4 = self.get_c(self.max_duration)
        self.s4 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + 3
        self.bytes5 = self.get_c(self.max_sessions * self.trial_types)
        self.s5 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + self.bytes5 + 4

    def get_c(self, val: int) -> int:
        if val == 1 or val == 0:
            return 1
        else:
            return int(math.ceil(math.log(val) / math.log(2)))

    def add_phase(self, seq_of_stimulus: str, is_random: bool, phase_num: int, is_configural_compounds: bool,
                  configural_compounds_mapping: Dict[str, str], timings: Any, iti: Any, context: Any, vartheta: bool) -> bool:
        # print(self.raw_stim)
        seq_of_stimulus = seq_of_stimulus.upper()
        self.add_trial_string(s=seq_of_stimulus)
        trial_set = []
        order = []
        sep = "/"
        listed_stimuli = seq_of_stimulus.split(sep)
        # print(listed_stimuli)
        csc_cues = None

        if context.get_symbol() not in self.cues and context.get_context() != config.Context:
            csc_cues = stimm.Stimulus(
                group=self, symbol=context.get_symbol(), trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
            self.cues[context.get_symbol()] = csc_cues

        no_stimuli = len(listed_stimuli)
        # print(no_stimuli)
        stimuli = {}

        configurals_added_to_this_grp = set()

        for i in range(no_stimuli):
            sel_stim = listed_stimuli[i]
            rep_stim = ""
            cues_name = ""
            stim_name = ""
            reinforced = False
            oktrials = False
            okcues = False

            if self.model.is_use_context():
                # print('JOY')
                cues_name = context.get_symbol()

            compound = ""
            no_stim_rep = 1

            for n in range(len(sel_stim)):
                sel_char = sel_stim[n]

                if sel_char.isdigit() and not oktrials:
                    rep_stim += sel_char
                elif sel_char.isalpha() and not okcues:
                    oktrials = True
                    cues_name += sel_char
                    if sel_char not in self.cues:
                        csc_cues = stimm.Stimulus(
                            group=self, symbol=sel_char, trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
                        self.cues[sel_char] = csc_cues
                        # print('wink', sel_char)
                    compound += sel_char
                    # print('=====>', compound)
                elif config.USNames.is_us(name=sel_char):
                    cues_name += sel_char
                    okcues = True
                    oktrials = True
                    reinforced = config.USNames.is_reinforced(name=sel_char)
                    # print(reinforced)
                    # print(sel_char)

                    if reinforced and sel_char not in self.cues:
                        us_cue = stimm.Stimulus(
                            group=self, symbol=sel_char, trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
                        self.cues[sel_char] = us_cue
                    elif not reinforced and not config.USNames.has_reinforced(names=sel_char):
                        us_cue = stimm.Stimulus(
                            group=self, symbol="+", trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
                        self.cues["+"] = us_cue
                else:
                    return False
                csc_cues = stimm.Stimulus(
                    group=self, symbol=compound, trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
                # self.cues[compound] = csc_cues

                if (self.model.is_use_context() or len(compound) > 1) and is_configural_compounds:
                    compound_set = set(compound)
                    # print('this is co set', compound_set)
                    if not self.model.is_serial_configurals():
                        for subset in self.power_set(compound_set):
                            # print('yo', subset)
                            s = "".join(sorted(subset))
                            s = self.model.is_use_context() and context.get_symbol() + s or s
                            # print('wanna see this ', s)
                            # print(len(s))
                            if len(s) > 1:

                                virtual_cue_name = self.get_key_by_value(
                                    configural_compounds_mapping, s)
                                if not virtual_cue_name:
                                    if not configural_compounds_mapping:
                                        virtual_cue_name = "a"
                                    else:
                                        char_val = [
                                            ord(list(k)[0]) for k in configural_compounds_mapping.keys()]
                                        char_val = chr(char_val[0] + 1)
                                        while not char_val.isalpha() or char_val.isupper() or config.ContextConfig.isContext(char_val):
                                            char_val = chr(ord(char_val) + 1)
                                        virtual_cue_name = char_val
                                    configural_compounds_mapping[virtual_cue_name] = s
                                cues_name += virtual_cue_name
                                # print('see this', cues_name)
                                configurals_added_to_this_grp.add(
                                    virtual_cue_name)
                                if virtual_cue_name not in self.cues:
                                    csc_cues = stimm.Stimulus(
                                        group=self, symbol=virtual_cue_name, trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
                                    self.cues[virtual_cue_name] = csc_cues
                                compound_name = s + virtual_cue_name
                                # print('cpd name', compound_name)
                                csc_cues = stimm.Stimulus(
                                    group=self, symbol=compound_name, trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
                                self.cues[compound_name] = csc_cues
            # print('cues', self.cues)
            # print('hala', halla.get())
            string_pos = sum(len(listed_stimuli[s]) + 1 for s in range(i))
            # print('string position', string_pos)
            # trial_num = sum(1 for s in range(i - 1, -1, -1)
            #                 if listed_stimuli[s] == listed_stimuli[i])
            trial_num = i+1
            # print(trial_num, 'trail num ')
            # print('cue names==string name in trals', cues_name)
            # print('selected stim', sel_stim)

            stim_name = cues_name
            # print('stim_name', stim_name)

            stim_temp = self.parseing(ttrial=stim_name, ctx=context)

            # print('selected_stim', sel_stim)
            trial = Trrial.Trial(trial_string=stim_name, sel_stim=sel_stim,
                                 string_pos=string_pos, trial_number=trial_num, sim_raw=self.raw_stim)

            # print(trial)
            trial_set.append(trial)
            if rep_stim:
                no_stim_rep = int(rep_stim)

            # print(list(stim_name), self.trial_strings, self.trial_strings2)
            # for stim_trial in stim_temp:

            if stim_name in stimuli:

                stimuli[stim_name].addTrials(no_stim_rep)
            else:
                # print('stim name', stim_name, 'reinformced', reinforced,
                #       'cuename', cues_name, 'num reprs', no_stim_rep)
                stimuli[stim_name] = Sim_Stim.SimStimulus(
                    name=stim_name, tr=no_stim_rep, cnames=cues_name, reinf=reinforced, raw_dta=self.raw_stim)
            # Append all trials in an order --->
            for _ in range(no_stim_rep):
                order.append(trial)
            # print(order)
            # print(stimuli)

        # print(hala.get())

        # self.create_common_cues()
        # self.add_common_stim_to_cues(common_dict=self.commons)

        # # print(len(order))
        # print('stimuli for simulation', stimuli)
        # print('cues', self.cues)
        # print('trials set', trial_set)
        for cue_names, cue_value in self.cues.items():
            for cue_names1, cue_value1 in self.cues.items():
                if cue_names1 in cue_value.names:
                    pass
                else:
                    cue_value.names.append(cue_names1)

                # if cue_names1   == '+':
                #     cue_value.names.append(cue_names1)

            # print(cue_names, cue_value.get_names())

        # print(hala.get())

        # timings.set_trials(len(order))
        # timings.restart_onsets()
        # sequences = timings.sequences(set(order))
        # if True:
        #     sequences = timings.compounds(list(order))

        # Have to find a way to get all the values in the model incuding the stim valuzz done

        self.has_random = is_random
        timings.restart_onsets()

        # for j in range(len(order)):
        #     timing = timings.make_timings(order[j].get_cues())
        # OBJECTIVE IS TO GET THE TIME SET DURATION (OFF - ON) AND SET TO RESPECTIVE STIM
        # SETTING MAX DURATION AS DURATIONN IF GRETAER
        self.set_timings(t_set=trial_set, timing_obj=timings)

        total_max = 0
        for j in range(len(order)):
            timing = timings.make_timings(order[j].get_cues())
            # print('===>', timing, iti.get_minimum())
            total_max = max(total_max, round(
                self.total_max + iti.get_minimum()))
        # print(total_max)

        self.max_duration = int(total_max)  # of the grp
        for s in self.cues.values():
            # print(s.get_name(), s.max_duration)
            if s.get_all_max_duration() < total_max:
                s.set_all_max_duration(int(total_max))
            # print(s.get_name(), s.get_all_max_duration(),  s.max_duration, s.total_max)

        names = list(self.cues.keys())
        # print(names)
        # print(halla.get())
        return self.add_phase_to_list(phas_num=phase_num, seq_stim=seq_of_stimulus, order=order, stimuli=stimuli, israndom=is_random, timngs=timings, iti_bj=iti, contxt=context)

    def add_phase_to_list(self, phas_num, seq_stim: str, order: List[Any], stimuli: Dict[str, Any], israndom: bool, timngs: Any, iti_bj: Any, contxt: Any):
        return self.phases.append(Phase.SimPhase(phaseNum=phas_num, sequence_of_stim=seq_stim, orderr=order, stimuli2=stimuli, grp=self, random=israndom, timing=timngs, iti_obj=iti_bj, ctx=contxt))

    def parseing(self, ttrial, ctx):

        stim = []
        if ttrial[0] == ctx.get_name():
            filterr1 = ttrial[1:-1]
            filterr2 = ttrial[-1]
            for i in filterr1:
                # print(i)
                stim.append(ttrial[0]+i+filterr2)
        else:
            filterr1 = ttrial[:-1]
            filterr2 = ttrial[-1]
            for i in filterr1:
                stim.append(i+filterr2)
        return stim

    def create_common_cues(self):
        for cue in self.cues.values():
            for cue2 in self.cues.values():
                if not cue.is_cs or not cue2.is_cs or cue2.get_name() == cue.get_name() or cue.is_common() or cue2.is_common():
                    continue
                if "c" + cue.get_name() + cue2.get_name() in self.commons or "c" + cue2.get_name() + cue.get_name() in self.commons:
                    continue
                csc_cues = stimm.Stimulus(
                    group=self, symbol="c" + cue.get_name() + cue2.get_name(),
                    trials=self.total_trials, total_stimuli=self.total_stimuli, corre_config=self.raw_stim)
                self.commons["c" + cue.get_name() + cue2.get_name()] = csc_cues

    def add_common_stim_to_cues(self, common_dict):
        for key in common_dict.keys():
            first = key[1]
            second = key[2]
            stim1 = self.cues.get(first)
            stim2 = self.cues.get(second)
            stim3 = common_dict.get(key)
            stim1.add_common(second, stim3)
            stim2.add_common(first, stim3)
            if key not in self.cues:
                self.cues[key] = stim3

    def set_timings(self, t_set, timing_obj):

        for j in range(len(t_set)):
            t_string = t_set[j].get_trial_string()
            timing = timing_obj.make_timings(t_set[j].get_cues())
            us_timings = us_timings = timing_obj.US_timings()
            total_max = timing_obj.t_max()

            for stim in self.cues.values():

                # if len(stim.get_name()) == 1 and not stim.is_context:

                names = list(stim.get_name())
                css = [None] * len(names)
                onset = -1
                offset = 200
                counter = 0
                for character in names:
                    # print('character', character)

                    # for cs in t_set[j].get_cues():
                    for cs in list(t_string):
                        # print(cs)
                        if cs == character:
                            # if cs.get_name() == character:
                            css[counter] = cs

                    if stim.is_cs or stim.is_context:

                        temp_onset = timing.get(
                            css[counter], [-1])[0] if css[counter] and css[counter] in timing else -1
                        onset = max(temp_onset, onset)
                        temp_offset = timing.get(
                            css[counter], [-1])[1] if css[counter] and css[counter] in timing else -1
                        offset = min(temp_offset, offset)
                        counter += 1

                    elif stim.is_us:

                        temp_onset = us_timings.get(
                            css[counter], [-1])[0] if css[counter] and css[counter] in us_timings else -1
                        onset = max(temp_onset, onset)
                        temp_offset = us_timings.get(
                            css[counter], [-1])[1] if css[counter] and css[counter] in us_timings else -1
                        offset = min(temp_offset, offset)
                        counter += 1

                stim.set_timing(onset, offset)

                duration = (
                    offset - onset) if not config.ContextConfig.isContext(stim.get_name()) else total_max

                # print(duration, stim.get_name(), stim.get_max_duration())
                if duration > stim.get_max_duration():
                    stim.set_max_duration(duration)
                if duration > total_max:
                    total_max = duration
                if stim.get_max_duration() > total_max:
                    total_max = stim.get_max_duration()

            # print('ist', stim.get_name(), stim.get_the_onset(),
                #   stim.get_the_offset(), stim.get_max_duration())

            # Set timings for new cues the CAB'zzz
            for s in self.cues.values():
                if len(s.get_name()) > 1:
                    onset = max(0, min(self.cues.get(s.get_name()[1]).get_the_onset(
                    ), self.cues.get(s.get_name()[2]).get_the_onset()))
                    offset = max(self.cues.get(s.get_name()[1]).get_the_offset(
                    ), self.cues.get(s.get_name()[2]).get_the_offset())
                    s.set_max_duration(offset - onset)
                    s.set_timing(onset, offset)
                s.set_all_max_duration(int(total_max))

                # print('2nd', s.get_name(), s.get_the_onset(),
                #       s.get_the_offset(), s.get_max_duration())

            for s in self.cues.values():
                if s.is_us and s.get_name() in config.USNames.get_names():
                    self.cues[s.get_name()].set_all_max_duration(
                        int(total_max))
                if s.is_us and s.get_name() in config.USNames.get_names() and (self.cues[s.get_name()].get_max_duration() < (us_timings[s.get_name()][1] - us_timings[s.get_name()][0])):

                    self.cues[s.get_name()].set_max_duration(
                        us_timings[s.get_name()][1] - us_timings[s.get_name()][0])

                    # onset = us_timings.get(s.get_name(), -1)[0]
                    # offset = us_timings.get(s.get_name(), -1)[1]
                    # s.set_timing(onset, offset)
                # print('3rd', s.get_name(), s.get_the_onset(),
                #       s.get_the_offset(), s.get_max_duration())
            max_micros = int(total_max)
            # print(max_micros)

    def power_set(self, original_set: Set[Any]) -> Set[frozenset]:
        sets = set()
        if not original_set:
            sets.add(frozenset())
            return sets
        list_set = list(original_set)
        head = list_set[0]
        rest = set(list_set[1:])
        for subset in self.power_set(rest):
            new_set = frozenset({head}.union(subset))
            sets.add(new_set)
            sets.add(subset)
        return sets

    def add_trial_string(self, s: str):
        strings = s.split("/")
        for s2 in strings:
            # print('s222', s2)
            # filtered = s2.replace(r'\d', '')
            filtered = re.sub(r'[\d]', '', s2)
            # print('f11', filtered)

            filtered2 = config.USNames.has_us_symbol(
                filtered) and filtered[:-1] or filtered
            # print('fil2', filtered2)

            if filtered2 not in self.trial_strings2:
                self.trial_strings2.append(filtered2)
            if filtered not in self.trial_strings:
                self.trial_strings.append(filtered)
        # print('=============>', self.trial_strings)

    def initialize_trial_arrays(self):
        for s in self.cues.values():
            s.initialize_trial_arrays()

    def get_key_by_value(self, config_cues_names: Dict[str, str], value: str) -> str:
        key = None
        count = 0
        if config_cues_names:
            for entry in config_cues_names.items():
                if entry[1] == value:
                    key = entry[0]
                    count += 1

        # print('aye', key)
        return key

    def get_cue_map(self):
        return self.cues

    def get_model(self):
        return self.model

    def clear_results(self):
        # print(self.phases)
        for x in range(self.no_of_phases):
            self.phases[x].results.clear()  # To clarifyyy

    # def get_first_occurrence(self, s: Any) -> int:
    #     print('not s', not s)
    #     if not s:
    #         return -1
    #     if s and s.get_name() in self.appearance_list:
    #         return self.appearance_list[s.get_name()]
    #     else:
    #         phase = -1
    #         for sp in self.phases:
    #             common_condition = s.is_common() and (s.a_stim and sp.contains_stimulus(s.a_stim)) or (s.b_stim and sp.contains_stimulus(s.b_stim))

    #             if phase == -1 and (sp.contains_stimulus(s) or common_condition or sp.get_context_config().get_symbol() == s.get_symbol()):
    #                 phase = sp.get_phase_num()
    #                 self.appearance_list[s.get_name()] = phase - 1

    #         return phase - 1

    def get_first_occurrence(self, s):
        if s is None:
            return -1

        if s.get_name() in self.appearance_list:
            return self.appearance_list[s.get_name()]
        else:
            phase = -1
            for sp in self.phases:
                # print('AYEEE', sp.containsStimulus(s), s.get_name())
                common_condition = s.is_common() and ((hasattr(s, 'a_stim') and s.a_stim is not None and sp.containsStimulus(s.a_stim)) or
                                                      (hasattr(s, 'b_stim') and s.b_stim is not None and sp.containsStimulus(s.b_stim)))
                # print(common_condition)
                if phase == -1 and (sp.containsStimulus(s) or common_condition or sp.get_context_config().get_symbol() == s.get_symbol()):

                    phase = sp.get_phase_num()
                    # print('yooooooo--->', phase)
                    self.appearance_list[s.get_name()] = phase - 1

            return sp.get_phase_num() - 1


# Useful in the model

    def set_lrate(self, rate: float):
        self.l_rate = rate

    def set_common(self, common: float):
        self.common = common

    def get_common(self) -> float:
        return self.common

    def get_lrate(self):
        return self.l_rate

    def get_phases(self):
        return self.phases

    def get_no_of_phases(self) -> int:
        return self.no_of_phases

    def get_no_trial_types(self) -> int:
        return len(self.trial_strings)

    def get_trial_type_index(self, s: str) -> int:
        return self.trial_strings.index(s)

    def get_trial_type_index2(self, s: str) -> int:
        return self.trial_strings2.index(s)


# Set at the notebook


    def set_total_trials(self, trials: int):
        self.total_trials = trials

    def get_total_trials(self) -> int:
        return self.total_trials

    def set_total_stimuli(self, stimuli: int):
        self.total_stimuli = stimuli

    def get_total_stimuli(self) -> int:
        return self.total_stimuli

# Run the model

    def run(self):
        # print(self.phases)
        self.add_microstimuli()
        for stim in self.cues.values():
            for se in stim.get_list():
                for stim_2 in self.cues.values():
                    for se_2 in stim_2.get_list():
                        if stim.get_name() in se_2.names:
                            pass
                        else:

                            se_2.names.append(stim.get_name())

        # for stim in self.cues.values():
            # for se in stim.get_list():
                # print(se.names)

        # print(halla.get())

        for i in range(len(self.phases)):
            # print(i)

            # if self.model.context_across_phase:
            #     for stim_name, stim_val in self.cues.items():
            #         name = self.model.get_config

            # if len(self.phases) > 1:
            #     continue  # TOdo

            # if i>0:
            #     for s in self.phases[i].get_cues().values():
            #         for s2 in self.phases[i - 1].get_cues().values():
            #             print(s2.get_list(), s2.get_name())
            #             if s.get_name() == s2.get_name():
            #                 for se in s.get_list():
            #                     for se2 in s2.get_list():
            #                         if se.get_microIndex() == se2.get_microIndex():
            #                             se.setVariableSalience(vs = se2.getVariableSalience())
            #                             se.setCSVariableSalience(vs = se2.getCSVariableSalience())

            self.phases[i].run()
            print('Executing===================>')

    # Functions to support running the model

    def reset_all(self):
        self.trial_strings = []
        self.appearance_list = defaultdict(int)
        self.compound_appearance_list = defaultdict(list)
        for s in self.cues.values():
            s.reset_completely()
        for sp in self.phases:
            sp.reset()

    def add_microstimuli(self):
        for stim in self.cues.values():
            stim.add_microstimuli()

            stim.setsize()

    def get_trial_strings(self) -> List[str]:
        return self.trial_strings

    def activate_test(self, sti: str):
        ele = {}
        for elemz in range(len(self.cues[sti].get_list())):
            ele[elemz] = self.cues[sti].get_list()[elemz].get_activation
        return ele
