"""
i16_peakfit.plotting
"""

import numpy as np
import matplotlib.pyplot as plt

# Setup matplotlib rc parameters
# These handle the default look of matplotlib plots
plt.rc('figure', figsize=[8, 6], dpi=100, autolayout=False)
plt.rc('figure.subplot', left=0.1, right=0.96, bottom=0.08, top=0.95, hspace=0.24, wspace=0.265)
plt.rc('lines', marker='o', color='r', linewidth=2, markersize=6)
plt.rc('errorbar', capsize=2)
plt.rc('legend', loc='best', frameon=False, fontsize=16)
plt.rc('axes', linewidth=2, titleweight='bold', labelsize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('axes.formatter', limits=(-3, 3), offset_threshold=6)
# Note font values appear to only be set when plt.show is called
plt.rc('font', family='serif', style='normal', weight='bold', size=12, serif=['Times New Roman', 'Times', 'DejaVu Serif'])
#plt.rcParams["savefig.directory"] = os.path.dirname(__file__) # Default save directory for figures
#plt.rcdefaults()


def plot_batch_results(values, results_list, value_name='value', title=None):
    """Plot amplitude, center, fwhm, background as figure"""
    # results_list = BatchGui.get_results()
    amplitude = [result['amplitude'] if 'amplitude' in result else np.nan for result in results_list]
    center = [result['center'] if 'center' in result else np.nan for result in results_list]
    fwhm = [result['fwhm'] if 'fwhm' in result else np.nan for result in results_list]
    background = [result['background'] if 'background' in result else np.nan for result in results_list]

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)
    fig.suptitle(title)

    ax1.plot(values, amplitude)
    ax1.set_xlabel(value_name)
    ax1.set_ylabel('Amplitude')

    ax2.plot(values, center)
    ax2.set_xlabel(value_name)
    ax2.set_ylabel('Centre')

    ax3.plot(values, fwhm)
    ax3.set_xlabel(value_name)
    ax3.set_ylabel('FWHM')

    ax4.plot(values, background)
    ax4.set_xlabel(value_name)
    ax4.set_ylabel('Background')

    plt.show()