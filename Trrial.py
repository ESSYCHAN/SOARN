import config
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import model
# import Raw_stim


# class CS:
#     def __init__(self, name: str, count: int, index: int, is_probe: bool, position: int):
#         self.name = name
#         self.count = count
#         self.index = index
#         self.is_probe = is_probe
#         self.position = position

#     def set_show_primes(self, show: bool):
#         self.show_primes = show

#     def __str__(self) -> str:
#         return f"CS{self.name, self.count,self.index, self.is_probe, self.position}"

#     def __repr__(self) -> str:
#         return f"CS{self.name, self.count,self.index, self.is_probe, self.position}"
#     # def get_name(self):
#     #     return self.name


# class ConfiguralCS(CS):
#     def __init__(self, name: str, count: int, index: int, additional_info: str, is_probe: bool):
#         super().__init__(name, count, index, is_probe, 0)
#         self.additional_info = additional_info


class Trial:
    def __init__(self, trial_string: str, sel_stim: str = None, string_pos: int = 0, trial_number: int = 0, sim_raw: dict = None):
        self.trial_string = trial_string
        # print(sim_raw)
        # print('this is trial string', self.trial_string)
        # print('this is selected stim', sel_stim)
        # self.is_probe = is_probe
        self.sim_raww = sim_raw
        self.cues = set()
        self.trial_number = trial_number
        # print(self.sim_raww)

        sel_stim = sel_stim or trial_string

        cues_with_configurals = sel_stim

        # Add in configurals
        for c in trial_string:
            if c.islower() or config.ContextConfig.isContext(c):
                cues_with_configurals += c

        cs = list(cues_with_configurals)
        # print(cs)
        probe_count: Dict[str, int] = {}
        compound = ""

        for i, c in enumerate(cs):
            # print(c)

            # print('starting index', string_index)
            # print(string_pos)
            if config.ContextConfig.isContext(c):

                # context = sim_raw[c]

                # context = sim_raw[c]
                # print(context)
                self.cues.add(sim_raw[c])
            elif c.isupper():

                count = probe_count.get(c, 0)
                if not self.timing_per_trial():
                    string_pos = 0
                    count = 0

                compound += c
                try:
                    probe_cs = cs[i + 1] == '^'
                    self.cues.add(
                        sim_raw[c])
                except IndexError:
                    self.cues.add(
                        sim_raw[c])

                count += 1
                probe_count[c] = count
            elif c.islower():
                self.cues.add(config.ConfiguralCS(c, 0, 0, "", False))
            # elif config.USNames.is_us(name=c):
            #     # print('CCCCC', c)
            #     if c == "+":
            #         self.cues.add(sim_raw[c])

            #     if c == "-":
            #         sim_raw["-"].set_offset(off=0)
            #         sim_raw["-"].set_onset(on=0)
            #         self.cues.add(sim_raw["-"])

            # if c.isalpha() and not config.ContextConfig.isContext(c) and self.timing_per_trial():
            #     string_index += 1
            string_pos += 1

        # print(self.cues)

        # Tell probe cues if they need to show primes
        # for cue in self.cues:
        #     if cue.is_probe:
        #         cue.set_show_primes(probe_count.get(cue.name, 0) > 1)

    # def __str__(self) -> str:
    #     pass

    # def __repr__(self) -> str:
    #     pass

    def copy(self):
        new_trial = Trial(trial_string=self.trial_string,
                          sim_raw=self.sim_raww)
        new_trial.set_cues(self.cues)
        new_trial.set_trial_number(self.trial_number)
        return new_trial

    def get_cues(self) -> Set[config.CS]:
        return self.cues

    def get_trial_string(self) -> str:
        return self.trial_string

    # def get_name(self):
    #     self

    # def is_probe(self) -> bool:
    #     return self.is_probe

    def set_cues(self, cues: Set[config.CS]):
        self.cues.clear()
        self.cues.update(cues)

    def set_probe(self, is_probe: bool):
        self.is_probe = is_probe

    def set_trial_string(self, trial_string: str):
        self.trial_string = trial_string

    def timing_per_trial(self) -> bool:
        return True
        # print('decision', model.SORN.get_timing_per_trial(self))
        # return model.SORN.get_timing_per_trial()
        # Assuming Simulator.get_controller().get_model().is_timing_per_trial() exists

        # return Simulator.get_controller().get_model().is_timing_per_trial()

    def __str__(self):
        return self.trial_string
    def __repr__(self):
        return self.trial_string

    def is_reinforced(self) -> bool:
        return self.trial_string.endswith("+")

    def get_probe_symbol(self) -> str:
        probe_name = "("
        for c in self.get_trial_string():
            if c.isupper():
                probe_name += c
        probe_name += "+" if self.is_reinforced() else "-"
        probe_name += ")"
        num_primes = self.trial_number
        primes = "'" * num_primes
        return probe_name + primes

    def get_trial_number(self) -> int:
        return self.trial_number

    def set_trial_number(self, trial_number: int):
        self.trial_number = trial_number
