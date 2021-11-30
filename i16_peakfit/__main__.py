"""
i16_peakfit
Wrapper and graphical user interface of lmfit for scattering experiments such as those on Diamond-I16.

Usage:
  $ python -m i16_peakfit
  > start gui
OR
  $ python -m bablescan /some/xye/file.dat
   - starts gui with data loaded

By Dan Porter, PhD
Diamond Light Source Ltd.
2021
"""
import i16_peakfit.functions

if __name__ == '__main__':

    import sys
    import i16_peakfit
    from i16_peakfit.tkinter_gui import FittingGUI

    print('\ni16_peakfit version %s, %s' % (i16_peakfit.__version__, i16_peakfit.__date__))
    print(' By Dan Porter, Diamond Light Source Ltd.')

    xdata = None
    ydata = None
    yerror = None
    for arg in sys.argv:
        if '.py' in arg:
            continue
        try:
            print('Opening: %s' % arg)
            xdata, ydata, yerror = i16_peakfit.functions.load_xye(arg)
        except Exception:
            pass

    FittingGUI(xdata, ydata, yerror)

