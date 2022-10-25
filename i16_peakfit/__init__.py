"""
i16_peakfit
Wrapper and graphical user interface of lmfit for scattering experiments such as those on Diamond-I16.

lmfit models are applied to scattering type peaks (Poisson-like using counting statistics where 0 is real).

Estimates are made as inital guesses based on additional peak finding algorithms.

Batch processing allows multiple datasets to be fitted at once and the resulting fit parameters plotted against
an ancilliary dataset value.

Usage:
Start GUI session:
  $ python -m i16_peakfit

OR - Start GUI with data loaded
  $ python -m i16_peakfit /some/xye/file.dat

OR - Import functions into script
    from i16_peakfit import multipeakfit, load_xye, peak_results_str

    xdata, ydata, yerror = load_xye('/some/data/file.txt')
    res = multipeakfit(xdata, ydata, yerror, npeaks=None, model='Gauss', plot_result=True)
    print(peak_results_str(res))

By Dan Porter, PhD
Diamond Light Source Ltd.
2021

Version 0.3.0
Last updated: 25/10/22

Version History:
16/11/21 0.1.0  Version History started.
30/11/21 0.2.0  Refactored tkinter_gui, functions.py, peak_finding.py, added tkmodelpars, tkbatch_gui, nexus_loader
25/10/22 0.3.0  Small updates before testing on beamline

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


__version__ = "0.3.0"
__date__ = "2022/10/25"

#import tkinter
#import matplotlib
#matplotlib.use("TkAgg")

from i16_peakfit.fitting import multipeakfit, generate_model, peak_results, peak_results_str, peak_results_plot
from i16_peakfit.peak_finding import find_peaks
from i16_peakfit.functions import load_xye, peak_ratio


def version_info():
    return 'i16_peakfit version %s (%s)' % (__version__, __date__)


def doc_str():
    return __doc__


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


