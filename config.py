from enum import Enum


class Context(Enum):
    EMPTY = 0


class ContextConfig:

    OMEGA = {}

    def __init__(self):
        self.symbol = '\u03A9'
        self.alpha_r = 0
        self.alpha_nr = 0
        self.salience = 0
        self.time = 0
        self.value1 = 0
        self.value2 = 0
        self.initialization()

    def initialization(self):
        ContextConfig.OMEGA[self.symbol] = self

    def get_symbol(self):
        return self.symbol

    def get_name(self):
        return self.symbol

    def get_alpha_r(self):
        return self.alpha_r

    def get_alpha_nr(self):
        return self.alpha_nr

    def get_salience(self):
        return self.salience

    def get_context(self):
        return self.symbol

    def get_name(self):
        return self.symbol

    def set_alpha_r(self, value):
        self.alpha_r = value

    def set_alpha_nr(self, value):
        self.alpha_nr = value

    def set_time(self, val):
        self.time = val

    def set_onset(self, on):
        self.value1 = on

    def set_offset(self, off):
        self.value2 = off

    def get_onset(self):
        return self.value1

    def get_offset(self):
        return self.value2

    @staticmethod
    def isContext(name):
        return name in ['\u03A9']

    def __getitem__(self, index):
        if index == 0:
            return self.value1
        elif index == 1:
            return self.value2
        else:
            raise ValueError

    # def __eq__(self, other):
    #     if isinstance(other, ContextConfig):
    #         return True
    #     else:
    #         return False

    def __str__(self):
        return f"Context{self.symbol, self.value1, self.value2}"

    def __repr__(self):
        return f"Context{self.symbol, self.value1, self.value2}"


class USNames:
    @staticmethod
    def is_us(name):
        return name in ["+", "-", "0"]

    @staticmethod
    def is_reinforced(name):
        return name == "+"

    @staticmethod
    def has_reinforced(names):
        return "+" in names

    @staticmethod
    def get_names():
        return ["+", "-", "0"]

    @staticmethod
    def has_us_symbol(names):
        return "+" in names or "-" in names


class CS:
    ALL = {}
    TOTAL = {}
    # OMEGA = {}
    US = {}

    def __init__(self, name: str, value1: float, value2: float, raw_data):
        self._value_ = name
        self.value1 = value1
        self.value2 = value2
        self.initialization()
        self.img = raw_data

    def initialization(self):
        if self._value_ == '\u03A9':
            raise ValueError(
                "Symbols should be one of CS's or US's use context config for context configuration")
            # CS.OMEGA[self._value_] = self
            # CS.ALL[self._value_] = self
        elif self._value_ in ["+", "-", "0"]:

            if self._value_ == '+':
                CS.US.clear()
                CS.US[self._value_] = self
                CS.ALL[self._value_] = self
            elif (self._value_ == '-') or (self._value_ == '0'):
                CS.US.clear()
                self.value1 = 0
                self.value2 = 0
                CS.US[self._value_] = self
                CS.ALL[self._value_] = self

            # else:
            #     raise ValueError(
            #         "Symbols should be one of these +, -, 0 for the US configuration")

        elif not USNames.is_us(self._value_) and not ContextConfig.isContext(self._value_):
            CS.TOTAL[self._value_] = self
            CS.ALL[self._value_] = self

    def __getitem__(self, index):
        if index == 0:
            return self.value1
        elif index == 1:
            return self.value2
        else:
            raise ValueError

    def __str__(self):
        return f"CS{self._value_, self.value1, self.value2}"

    def __repr__(self):
        return f"CS{self._value_, self.value1, self.value2}"

    def get_name(self) -> str:
        return self._value_

    def get_symbol(self):
        return self._value_

    def set_onset(self, on):
        self.value1 = on

    def get_context(self):
        return self._value_

    def set_offset(self, off):
        self.value2 = off

    def get_onset(self):
        return self.value1

    def get_offset(self):
        return self.value2

    def set_alpha_r(self, value):
        self.alpha_r = value

    def set_alpha_nr(self, value):
        self.alpha_nr = value

    def get_alpha_r(self):
        return self.alpha_r

    def get_alpha_nr(self):
        return self.alpha_nr

    def set_beta(self, value):
        self.beta = value

    def get_beta(self):
        return self.beta

    def set_salience(self, value):
        self.salience = value

    def get_salience(self):
        return self.salience

    def set_alpha_plus(self, value):
        self.alpha_plus = value

    def get_alpha_plus(self):
        return self.alpha_plus

    @staticmethod
    def isContext(name):
        return name in ['\u03A9']

    @staticmethod
    def isUS(name):
        return name in ["+", "-", "0"]


class ConfiguralCS(CS):
    SERIAL_SEP = "\u2192"

    def __init__(self, name, value1, value2,  parts="", serial=False):
        super().__init__(name, value1, value2)
        self.parts = parts
        self.serial = serial
        self.initn()

    def initn(self):
        #         sep =
        if self.serial == True:
            j = self.get_parts().split(",")
            print('aye', j)
            self._value_ = j[0] + ConfiguralCS.SERIAL_SEP + j[1]
            CS.TOTAL = self
        else:
            CS.TOTAL = self

    def get_parts(self):
        return self.parts

    def is_serial_configural(self):
        return self.serial

    def set_parts(self, parts):
        self.parts = parts
