
import numpy as np
from collections import defaultdict, OrderedDict
from typing import List, Dict, Set, Any
import scipy.spatial.distance as dist
import config
import Group


class SORN:
    def __init__(self):
        self.use_context = None
        self.serial_configurals = False
        self.timing_per_trial = False
        self.values: Dict[str, float] = {}
        self.usCV = 1
        self.csCV = 1
        self.groups = {}
        self.listAllCues = []
        self.randomPhases: int = 0
        self.phasesNo = 0
        self.groupsNo = 0
        self.reset_context: bool = True
        self.skew: float = 5
        self.reset_value: float = 0.95
        self.persistence: int = 0
        self.commonMap: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.stimNames: List[str] = []
        self.combinationNo = 1
        self.isErrors = False
        self.isErrors2 = False
        self.context_across_phase = True

    def addCueNames(self, mapp):
        for pair in mapp.items():
            self.values[pair[1].get_symbol()] = 0.0
            if pair[1].get_symbol() not in self.listAllCues:
                self.listAllCues.append(pair[1].get_symbol())
        # print(self.listAllCues)

    def addGroupIntoMap(self, name: str, group):
        self.groups[name] = group
        self.addCueNames(group.get_cue_map())
        # self.randomPhases += group.numRandom() #ToDo

    # to set in the notebook before initilialization

    def set_GroupNo(self, g: int):
        self.groupsNo = g

    def set_PhaseNo(self, p: int):
        self.phasesNo = p

    def set_learning_rate(self, i):
        self.learning_rate = i

    def set_reset_context(self, r: bool):
        self.reset_context = r

    def set_cs_cv(self, n):
        self.csCV = n

    def set_us_cv(self, n):
        self.usCV = n

    def set_persistance(self, n):
        self.persistence = n

    def set_skew(self, val):
        self.skew = val

    def set_reset_value(self, r):
        self.reset_value = r

    def set_size(self, n):
        self.size = n

    # set kwa notebook
    def set_cs_scalar(self, n):
        self.cs_scalar = n

    def set_us_scalar(self, n):
        self.us_scalar = n

    def set_Intensity(self, i):
        self.intensity = i

    def set_is_Errros(self, b1: bool, b2: bool):
        self.isErrors = b1
        self.isErrors2 = b2

    def setExternalSave(self, isExternalSave: bool):
        self.isExternalSave = isExternalSave

    def get_isExternalSave(self) -> bool:
        return self.isExternalSave

    def run(self):
        self.startCalculations()

    # def startCalculations(self):
    #     self.listAllCues.clear()
    #     self.groupPool = [for _ in range(self.groupsNo)]
    #     print(self.groupPool)

    def get_PhaseNo(self):
        return self.phasesNo

    def get_GroupNo(self):
        return self.groupsNo

    def get_skew(self):
        return self.skew

    def get_reset_value(self):
        return self.reset_value

    def get_discount(self):
        return self.values['discount']

    def get_persistance(self):
        return self.persistence

    def get_cs_cv(self):
        return self.csCV

    def get_us_cv(self):
        return self.usCV

    def get_learning_rate(self):
        return self.learning_rate

    def get_size(self):
        return self.size

    def get_cs_scalar(self):
        return self.cs_scalar

    def get_us_scalar(self):
        return self.us_scalar

    def get_Intensity(self):
        return self.intensity

    def get_isErrors(self):
        return self.isErrors

    def get_isErrors2(self):
        return self.isErrors2


# Setting propotions


    def initializeCommonMap(self):
        if not self.commonMap:
            common = 30
            # otm = self.view.getOtherValuesTableModel()
            # common = float(otm.getValueAt(2, 1))

            for sg in self.groups.values():
                self.stimNames = []
                self.commonMap[sg.get_name_of_grp()] = {}
                for stim in sg.get_cue_map().values():
                    # print('==>', stim.get_name())
                    if stim.get_name() not in self.stimNames and stim.is_common():
                        self.stimNames.append(stim.get_name())
                        self.commonMap[sg.get_name_of_grp()
                                       ][stim.get_name()] = self.sim_id(name_stim=stim.get_name())

            # print(self.stimNames)
            # print(self.commonMap)
            # selr

            # otm = self.view.getOtherValuesTableModel()

        # print('after init', self.commonMap)

    def sim_id(self, name_stim):
        stimz = []
        for names in name_stim:
            if names == 'c':
                pass
            else:
                for sg in self.groups.values():
                    for stimkey, stim_val in sg.get_cue_map().items():
                        if names == stimkey:
                            stimz.append(
                                stim_val.raw_data[names].img.filter_maps)
        comon = self.clac_similarity_index(img1=stimz[0], img2=stimz[0])
        return comon

    def clac_similarity_index(self, img1, img2):
        AA = set(img1.flatten())
        BB = set(img2.flatten())
        common = len(AA.intersection(BB))
        # print('shared', common)
        return common
        # print(dist.euclidean(img1.flatten(), img2.flatten()))

    def getCombinationNo(self) -> int:
        return self.combinationNo

    def setCombinationNo(self, r: int):
        self.combinationNo = r

    def setCommonMap(self, common: Dict[str, Dict[str, float]]):
        if common is not None:
            self.commonMap = common
        self.calculateCommonProportions()

    def getCommon(self) -> Dict[str, Dict[str, float]]:
        return self.commonMap

    def getProportions(self) -> Dict[str, Dict[str, float]]:
        return self.proportions

    def calculateCommonProportions(self):
        self.proportions = {}
        for groupName in self.groups.keys():
            self.proportions[groupName] = {}
        for groupName in self.groups.keys():
            for stimName in self.groups[groupName].get_cue_map().keys():
                total = 0.0
                for commonName in self.commonMap[groupName].keys():
                    if commonName in stimName:
                        total += self.commonMap[groupName][commonName]
                self.proportions[groupName][stimName] = total

        for groupName in self.proportions.keys():
            for stimName in self.proportions[groupName].keys():
                if self.proportions[groupName][stimName] >= 1:
                    # print('====>', stimName)
                    for commonName in self.stimNames:
                        if commonName in stimName:
                            self.commonMap[groupName][commonName] /= self.proportions[groupName][stimName]
                            # print('FINAL_MAP', self.commonMap)

    def is_use_context(self):
        return self.use_context

    def is_serial_configurals(self):
        return self.serial_configurals

    def set_is_use_context(self, ctx: bool):
        self.use_context = ctx

    def set_is_serial_configurals(self, scc: bool):
        return self.serial_configurals

    def set_timing_per_trial(self, decision: bool):
        self.timing_per_trial = decision

    def get_timing_per_trial(self):
        return self.timing_per_trial

    def get_alpha_cues(self) -> Dict[str, float]:
        tm = {}
        for pair in self.values.keys():
            if "\u03BB" not in pair and "\u03B1" not in pair:
                tm[pair] = self.values.get(pair)
        return tm

    def update_values(self, name: str, phase: int, value: str):
        usNames = []
        alphaPlusNames = []
        BetaNames = []
        omicronNames = []  # Salience
        usLambdaNames = []
        for s in self.listAllCues:
            if config.USNames.is_us(name=s) and s not in usNames:
                usNames.append(s)
                alphaPlusNames.append(f"{s} - \u03B1+")
                BetaNames.append(f"{s} - \u03B2")
                omicronNames.append(f"{s}_s")
                usLambdaNames.append(f"{s} - \u03BB")

        if value == "":
            isAlphaplus = False
            for alpha_p_names in alphaPlusNames:
                if alpha_p_names in name:
                    self.values[name +
                                f"p{phase}"] = self.values.get(f"{alpha_p_names} p1")
                    isAlphaplus = True
                elif "\u03B1+" in name:
                    self.values[name +
                                f"p{phase}"] = self.values.get(" \u03B1+ p1")
                    isAlphaplus = True

            isBeta = False
            for beta_names in beta_names:
                if beta_names in name:
                    self.values[name +
                                f"{phase}"] = self.values.get(f"{beta_names} p1")
                    isBeta = True
                elif "\u03B2" in name:
                    self.values[name +
                                f"{phase}"] = self.values.get("\u03B2 p1")
                    isBeta = True

            isOmicron = False
            for omicron_names in omicron_names:
                if omicron_names in name:
                    self.values[name +
                                f"p{phase}"] = self.values.get(f"{omicron_names} p1")
                    isOmicron = True
                elif "+_s" in name:
                    self.values[name + f"p{phase}"] = self.values.get("+_s p1")
                    isOmicron = True
            isLambda = False

            for lamda_names in usLambdaNames:
                if lamda_names in name:
                    self.values[name +
                                f"{phase}"] = self.values.get(f"{lamda_names} p1")
                    isLambda = True
                elif "\u03BB" in name:
                    self.values[name +
                                f"{phase}"] = self.values.get("\u03BB p1")
                    isLambda = True

            if not (isAlphaplus or isBeta or isOmicron or isLambda):  # AKA if all of them are not set

                if "gamma" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("gamma p1")
                elif "CS \u03C1 " in name:
                    self.values[name +
                                f"{phase}"] = self.values.get("intergration p1")
                elif "US \u03C1" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("US \u03C1 p1")

                elif "Threshold" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("Threshold p1")

                elif "Variable Salience" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("Variable Salience p1")
                elif "skew" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("skew p1")
                elif "\u03C6" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("\u03C6 p1")
                elif "Wave Constant" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("Wave Constant p1")
                elif "US Scalar Constant" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("US Scalar Constant p1")
                elif "delta" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("delta p1")
                elif "b" in name:
                    self.values[name + f" p{phase}"] = self.values.get("b p1")
                elif "common" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("common p1")
                elif "setsize" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("setsize p1")
                elif "\u03C2" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("\u03C2 p1")
                elif "CV" in name:
                    self.values[name + f" p{phase}"] = self.values.get("CV p1")
                elif "linear c" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("linear c p1")
                elif "\u03c41" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("\u03c41 p1")
                elif "\u03c42" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("\u03c42 p1")
                elif "Salience Weight" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("Salience Weight p1")
                elif "\u03d1" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("\u03d1 p1")
                elif "CS \u03C1" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("CS \u03C1 p1")
                elif "Self Discount" in name:
                    self.values[name +
                                f" p{phase}"] = self.values.get("Self Discount p1")
                elif "ω" in name:
                    self.values[name + f" p{phase}"] = self.values.get("ω p1")
                else:
                    self.values[name + f" p{phase}"] = float(value)

        else:
            self.values[name] = float(value)
            self.values[name + f"p{phase}"] = float(value)

        # print(self.values)

    # def set_lrate(self, i):
    #     self.learning_rate = i

    def update_values_on_grps(self):
        for grp_name, grp_obj in self.groups.items():
            grp_obj.clear_results()

            for cue_keys, temp_csc_cues in grp_obj.get_cue_map().items():
                # print('stim--->', cue_keys)
                temp_csc_cues.reset(False, 0)
                alpha_r_value = self.values.get(
                    f"{temp_csc_cues.get_name()} - \u03B1r", -1)
                # print(alpha_r_value)
                alpha_n_value = self.values.get(
                    f"{temp_csc_cues.get_name()} - \u03B1n", -1)
                # print('this is alphan ', alpha_n_value)
                salience = self.values.get(f"{temp_csc_cues.get_name()}_s", -1)
                # print(salience, 'salience')

                if len(temp_csc_cues.get_name()) > 1:
                    # print('dnfjsnsj')
                    char = list(temp_csc_cues.get_name())
                    for ch in char:
                        # print(ch)
                        if ch != 'c':
                            alpha_r_value += self.values.get(
                                f"{ch} - \u03B1r", 0)
                            # print('this alspha r', alpha_r_value)
                            alpha_n_value += self.values.get(
                                f"{ch} - \u03B1n", 0)
                    alpha_r_value /= len(temp_csc_cues.get_name())
                    alpha_n_value /= len(temp_csc_cues.get_name())
                    # print('fsnf', alpha_n_value, alpha_r_value)

                    if alpha_r_value != -1 and alpha_r_value != 0:
                        temp_csc_cues.set_r_alpha(alpha=alpha_r_value)
                    if alpha_n_value != -1 and alpha_n_value != 0:
                        temp_csc_cues.set_n_alpha(alpha=alpha_n_value)
                    if salience not in (-1, 0):
                        temp_csc_cues.set_salience = salience

                if alpha_r_value != -1 and alpha_r_value != 0:
                    temp_csc_cues.set_r_alpha(alpha=alpha_r_value)
                if alpha_n_value != -1 and alpha_n_value != 0:
                    temp_csc_cues.set_n_alpha(alpha=alpha_n_value)
                if salience not in (-1, 0):
                    temp_csc_cues.set_salience = salience

            us_Names = []
            us_alpha_Names = []
            us_beta_Names = []
            us_salience_Names = []

            for s in self.listAllCues:
                if config.USNames.is_us(name=s) and s not in us_Names:
                    us_Names.append(s)
                    us_alpha_Names.append(f"{s} - \u03B1+")
                    us_beta_Names.append(f"{s} - \u03B2")
                    us_salience_Names.append(f"{s}_s")

            for i in range(1, self.phasesNo+1):
                grp_obj.set_lrate(self.learning_rate)

                for alpha_name in us_alpha_Names:
                    key = alpha_name
                    # print('let me see the key', key)
                    if key in self.values:
                        grp_obj.get_cue_map()[alpha_name.split(
                            ' -')[0]].set_r_alpha(alpha=self.values[key])

                for beta_name in us_beta_Names:
                    key = beta_name

                    # print('let me see the key', key)
                    if key in self.values:
                        grp_obj.get_cue_map()[beta_name.split(
                            ' - ')[0]].set_beta = self.values[key]

                for salience_name in us_salience_Names:
                    key = salience_name
                    if key in self.values:
                        grp_obj.get_cue_map()[salience_name.split(
                            '_')[0]].set_salience = self.values[key]

                if f"gammap{i}" in self.values:
                    grp_obj.get_phases()[
                        i-1].set_gamma(gma=self.values[f"gammap{i}"])

                if f"Wave_kp{i}" in self.values:
                    grp_obj.get_phases()[
                        i-1].set_cs_scalar(value=self.values[f"Wave_kp{i}"])

                if f"ϑp{i}" in self.values:
                    # print('lol', self.values[f"ϑp{i}"])
                    grp_obj.get_phases()[
                        i-1].set_vartheta(v=self.values[f"ϑp{i}"])

                if f"US \u03C1p{i}" in self.values:  # 'ρ'
                    grp_obj.get_phases()[
                        i-1].set_leak(us=self.values[f"US \u03C1p{i}"], cs=self.values[f"CS \u03C1p{i}"])

                if f"deltap{i}" in self.values:
                    grp_obj.get_phases()[
                        i-1].set_delta(delta=self.values[f"deltap{i}"])

                if f"discountp{i}" in self.values:
                    grp_obj.get_phases()[
                        i-1].set_self_pred(d=self.values[f"discountp{i}"])

                if f"ωp{i}" in self.values:
                    grp_obj.get_phases()[
                        i-1].set_context_salience(salience=self.values[f"ωp{i}"])

                if f"bp{i}" in self.values:
                    for stim in grp_obj.get_cue_map().values():
                        stim.set_b(b=self.values[f"bp{i}"])

                if f"commonp{i}" in self.values:
                    grp_obj.set_common(common=self.values[f"commonp{i}"])
                # print(grp_obj.get_phases()[i-1])
                grp_obj.get_phases()[
                    i-1].set_reset_context(r=self.reset_context)
                grp_obj.get_phases()[i-1].set_std(s=self.get_cs_cv())
                grp_obj.get_phases()[i-1].set_us_std(s=self.get_us_cv())
                grp_obj.get_phases()[
                    i-1].set_us_persistance(p=self.get_persistance())
                grp_obj.get_phases()[
                    i-1].set_cs_c_like(max(0, self.get_skew()))
                grp_obj.get_phases()[
                    i-1].set_val_context_reset(self.get_reset_value())
                # grp_obj.get_phases()[i-1].set_subset_size(self.get_size())
                grp_obj.get_phases()[i-1].set_intensity(i=self.get_Intensity())
                grp_obj.get_phases()[i-1].set_parameters()

                # print(self.values)
