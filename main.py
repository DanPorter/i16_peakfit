"""
i16_peakfit
Wrapper and graphical user interface of lmfit for scattering experiments such as those on Diamond-I16.

Usage:
  $ python -m i16_peakfit
  > start gui
OR
  $ python -m i16_peakfit /some/xye/file.dat
   - starts gui with data loaded

By Dan Porter, PhD
Diamond Light Source Ltd.
2021
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import i16_peakfit
from i16_peakfit.tkinter_gui import FittingGUI


if __name__ == '__main__':
    gui = FittingGUI()

