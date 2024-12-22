import config
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import numpy as np


class SimStimulus:
    def __init__(self, name: str, tr: int, cnames: str, reinf: bool, raw_dta: np.array):
        self.fullName = name
        self.trials = tr
        self.cueNames = cnames
        self.reinforced = reinf
        self.partsString = []
        self.parts: List[config.CS] = []
        # print('---->', raw_dta)
        self.size = 0
        for names in cnames:
            # print("i want to see these names", names)

            # self.addPart(config.CS(c, 0, 0, raw_data=raw_dta[c]))
            if 'c' in names:  # know how to handle this
                pass
                # print(names[1], names[2])
            elif names == '\u03A9':
                self.addPart(raw_dta[names])
                self.size += 1
            elif names == "-" or names == '+':
                self.addPart(raw_dta['+'])
                self.size += raw_dta['+'].img.filter_maps.size
            # else:

            #     self.size += raw_dta[names].img.filter_maps.size

    def addPart(self, part: config.CS):
        self.parts.append(part)
        self.partsString.clear()
        # pr
    # def __str__(self) -> str:
    #     pass
    def addTrials(self, n: int):
        self.trials += n

    def contains(self, part: config.CS) -> bool:
        return part in self.parts or part.getName() == "*"

    def contains(self, parts: str) -> bool:
        if "(" in parts:
            bits = parts.split("(")
            csTrialString = bits[1].split(")")[0]
            if csTrialString != self.fullName:
                return False
            parts = bits[0].split("'")[0]

        if self.contains(config.CS(parts, 0, 0)):
            return True

        for c in parts:
            if c not in self.getPartsString():
                return False
        return True

    def getCueNames(self) -> str:
        return self.cueNames

    def getName(self) -> str:
        return self.fullName

    def getParts(self) -> List[config.CS]:
        return self.parts

    def getPartsString(self) -> List[str]:
        if not self.partsString:
            self.partsString = [c.getName() for c in self.parts]
        return self.partsString

    def getTrials(self) -> int:
        return self.trials

    def isReinforced(self) -> bool:
        return self.reinforced

    def setParts(self, parts: List[config.CS]):
        self.parts = parts

   

