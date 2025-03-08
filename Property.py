
class Body_Figure(object):
    def __str__(self):
        return f"Body Figure Information:\n"\
                f" - Waist-to-Shoulder Ratio (WSR): {self.WSR}\n"\
               f" - Waist-to-Thigh Ratio (WTR): {self.WTR}\n"\
               f" - Waist-to-Hip Ratio (WHpR): {self.WHpR}\n"\
               f" - Waist-to-Head Ratio (WHdR): {self.WHdR}\n"\
               f" - Hip-to-Head Ratio (HpHdR): {self.HpHdR}\n"\
               f" - Area: {self.Area}\n"\
               f" - Height-to-Waist Ratio (H2W): {self.H2W}\n"

               
    def __init__(self, waist_width, thigh_width, hip_width, head_width, Area, height, shoulder_width):
        self._waist_width = waist_width
        self._thigh_width = thigh_width
        self._hip_width = hip_width
        self._head_width = head_width
        self._Area = Area
        self._height = height
        self._shoulder_width = shoulder_width
        if self._head_width == 0:
            self._head_width = self._hip_width/3

    @property
    def WSR(self):
        return (self._waist_width) / (self._shoulder_width)

    @property
    def WTR(self):
        return (self._waist_width / self._thigh_width)  # **2

    @property
    def WHpR(self):
        return (self._waist_width / self._hip_width)  # **2

    @property
    def WHdR(self):
        return (self._waist_width / self._head_width)  # **2

    @property
    def HpHdR(self):
        return (self._hip_width / self._head_width)  # **2

    @property
    def Area(self):
        return self._Area

    @property
    def H2W(self):
        return self._height / self._waist_width