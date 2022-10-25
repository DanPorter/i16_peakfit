"""
i16_peakfit Peak Finding functions
"""

import numpy as np

from i16_peakfit.functions import error_func, group_adjacent


def local_maxima_1d(y):
    """
    Find local maxima in 1d array
    Returns points with central point higher than neighboring points
    Copied from scipy.signal._peak_finding_utils
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/_peak_finding_utils.pyx
    :param y: list or array
    :return: array of peak indexes
    """
    y = np.asarray(y, dtype=float).reshape(-1)

    # Preallocate, there can't be more maxima than half the size of `y`
    midpoints = np.empty(y.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays
    i = 1  # Pointer to current sample, first one can't be maxima
    i_max = y.shape[0] - 1  # Last sample can't be maxima
    while i < i_max:
        # Test if previous sample is smaller
        if y[i - 1] < y[i]:
            i_ahead = i + 1  # Index to look ahead of current sample

            # Find next sample that is unequal to x[i]
            while i_ahead < i_max and y[i_ahead] == y[i]:
                i_ahead += 1

            # Maxima is found if next unequal sample is smaller than x[i]
            if y[i_ahead] < y[i]:
                left_edge = i
                right_edge = i_ahead - 1
                midpoints[m] = (left_edge + right_edge) // 2
                m += 1
                # Skip samples that can't be maximum
                i = i_ahead
        i += 1
    return midpoints[:m]


def find_local_maxima(y, yerror=None):
    """
    Find local maxima in 1d arrays, returns index of local maximums, plus
    estimation of the peak power for each maxima and a classification of whether the maxima is greater than
    the standard deviation of the error
    E.G.
        index, power, isgood = find_local_maxima(ydata)
        maxima = ydata[index[isgood]]
        maxima_power = power[isgood]
    Peak Power:
      peak power for each maxima is calculated using the peak_ratio algorithm for each maxima and adjacent points
    Good Peaks:
      Maxima are returned Good if:  power > (max(y) - min(y)) / std(yerror)
    :param y: array(n) of data
    :param yerror: array(n) of errors on data, or None to use default error function (sqrt(abs(y)+1))
    :return index: array(m<n) of indexes in y of maxima
    :return power: array(m) of estimated peak power for each maxima
    :return isgood: bool array(m) where True elements have power > power of the array
    """

    if yerror is None or np.all(np.abs(yerror) < 0.1):
        yerror = error_func(y)
    yerror[yerror < 1] = 1.0
    bkg = np.min(y)
    wi = 1 / yerror ** 2

    # Find points of inflection
    index = local_maxima_1d(y)
    # average nearest 3 points to peak
    power = np.array([np.sum(wi[m - 1:m + 2] * (y[m - 1:m + 2] - bkg)) / np.sum(wi[m - 1:m + 2]) for m in index])
    # Determine if peak is good
    isgood = power > (np.max(y) - np.min(y)) / (np.std(yerror) + 1)
    return index, power, isgood


def find_peaks(y, yerror=None, min_peak_power=None, points_between_peaks=6):
    """
    Find peak shaps in linear-spaced 1d arrays with poisson like numerical values
    E.G.
      index, power = find_peaks(ydata, yerror, min_peak_power=None, peak_distance_idx=10)
      peak_centres = xdata[index]  # ordered by peak strength
    :param y: array(n) of data
    :param yerror: array(n) of errors on data, or None to use default error function (sqrt(abs(y)+1))
    :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
    :param points_between_peaks: int, group adjacent maxima if closer in index than this
    :return index: array(m) of indexes in y of peaks that satisfy conditions
    :return power: array(m) of estimated power of each peak
    """
    # Get all peak positions
    midpoints, peak_signals, chk = find_local_maxima(y, yerror)

    if min_peak_power is None:
        good_peaks = chk
    else:
        good_peaks = peak_signals >= min_peak_power

    # select indexes of good peaks
    peaks_idx = midpoints[good_peaks]
    peak_power = peak_signals[good_peaks]
    if len(peaks_idx) == 0:
        return peaks_idx, peak_power

    # Average peaks close to each other
    group_idx, group_signal_idx = group_adjacent(peaks_idx, points_between_peaks)
    peaks_idx = np.round(group_idx).astype(int)
    peak_power = np.array([np.sum(peak_power[ii]) for ii in group_signal_idx])

    # sort peak order by strength
    power_sort = np.argsort(peak_power)
    return peaks_idx[power_sort], peak_power[power_sort]


def pixel_peak_search(data, peak_percentile=99):
    """
    find average position of bright points in image
    :param data: numpy array with ndims 1,2,3
    :param peak_percentile: float from 0-100, percentile of image to use as peak area
    :return: i, j, k index of image[i,j,k]
    """
    # bright = data > (peak_percentile/100.) * np.max(image)
    bright = data > np.percentile(data, peak_percentile)
    weights = data[bright]

    if np.ndim(data) == 3:
        shi, shj, shk = data.shape
        j, i, k = np.meshgrid(range(shj), range(shi), range(shk))
        avi = np.average(i[bright], weights=weights)
        avj = np.average(j[bright], weights=weights)
        avk = np.average(k[bright], weights=weights)
        return int(avi), int(avj), int(avk)
    elif np.ndim(data) == 2:
        shi, shj = data.shape
        j, i = np.meshgrid(range(shj), range(shi))
        avi = np.average(i[bright], weights=weights)
        avj = np.average(j[bright], weights=weights)
        return int(avi), int(avj)
    elif np.ndim(data) == 1:
        i = np.arange(len(data))
        avi = np.average(i[bright], weights=weights)
        return int(avi)
    else:
        raise TypeError('wrong data type')

