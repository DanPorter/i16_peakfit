"""
i16_peakfit
Wrapper and graphical user interface of lmfit for scattering experiments such as those on Diamond-I16.

By Dan Porter, PhD
Diamond Light Source Ltd.
2021

Version 1.0.0
Last updated: 16/11/21

Version History:
16/11/21 1.0.0  Version History started.

-----------------------------------------------------------------------------
   Copyright 2021 Diamond Light Source Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 Dr Daniel G Porter, dan.porter@diamond.ac.uk
 www.diamond.ac.uk
 Diamond Light Source, Chilton, Didcot, Oxon, OX11 0DE, U.K.
"""


__version__ = "1.0.0"
__date__ = "2021/11/16"

#import tkinter
#import matplotlib
#matplotlib.use("TkAgg")

from i16_peakfit.fitting import load_xye, find_peaks, peak_ratio, multipeakfit, generate_model


def version_info():
    return 'i16_peakfit version %s (%s)' % (__version__, __date__)


def module_info():
    import sys
    out = 'Python version %s' % sys.version
    out += '\n at: %s' % sys.executable
    out += '\n %s: %s' % (version_info(), __file__)
    # Modules
    import numpy
    out += '\n     numpy version: %s' % numpy.__version__
    import lmfit
    out += '\n      h5py version: %s' % lmfit.__version__
    import matplotlib
    out += '\nmatplotlib version: %s' % matplotlib.__version__
    import tkinter
    out += '\n   tkinter version: Tk: %s, Tcl: %s' % (tkinter.TkVersion, tkinter.TclVersion)
    import os
    out += '\nRunning in directory: %s\n' % os.path.abspath('.')
    return out


