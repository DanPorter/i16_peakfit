"""
i16_peakfit Example script to fit x/y data
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
from i16_peakfit import fitting, load_xye

r"""
HdfScan(filename: C:\Users\dgpor\Dropbox\Python\ExamplePeaks\794940.nxs, namespace: 54, associations: 15)
        scan_command = scan eta 29.711 30.311 0.01 BeamOK pil3_100k 1 roi2
                 eta = (61,) max: 30.311, min: 29.711, mean: 30.011
            roi2_sum = (61,) max: 3.4933e+06, min: 5334, mean: 6.5055e+05
"""
xdata, ydata, yerror = load_xye('example_peaks/Scan_794940_eta_roi2_sum.csv')
# xdata, ydata, yerror = np.loadtxt('example_peaks/Scan_794940_eta_roi2_sum.csv', delimiter=',').T

mod, pars = fitting.generate_model(xdata, ydata, yerror)

res = fitting.multipeakfit(xdata, ydata, yerror)

print(fitting.peak_results_str(res))

fig = fitting.peak_results_plot(res, xlabel='eta', ylabel='roi2_sum', title='794940.nxs')

