"""
i16_peakfit general functions
"""

import sys, os
import re
import json
import numpy as np


def shortstr(string):
    """
    Shorten string by removing long floats
    :param string: string, e.g. '#810002 scan eta 74.89533603616637 76.49533603616636 0.02 pil3_100k 1 roi2'
    :return: shorter string, e.g. '#810002 scan eta 74.895 76.495 0.02 pil3_100k 1 roi2'
    """
    #return re.sub(r'(\d\d\d)\d{4,}', r'\1', string)
    def subfun(m):
        return str(round(float(m.group()), 3))
    return re.sub(r'\d+\.\d{5,}', subfun, string)


def stfm(val, err):
    """
    Create standard form string from value and uncertainty"
     str = stfm(val,err)
     Examples:
          '35.25 (1)' = stfm(35.25,0.01)
          '110 (5)' = stfm(110.25,5)
          '0.0015300 (5)' = stfm(0.00153,0.0000005)
          '1.56(2)E+6' = stfm(1.5632e6,1.53e4)

    Notes:
     - Errors less than 0.01% of values will be given as 0
     - The maximum length of string is 13 characters
     - Errors greater then 10x the value will cause the value to be rounded to zero
    """

    # Determine the number of significant figures from the error
    if err == 0. or val / float(err) >= 1E5:
        # Zero error - give value to 4 sig. fig.
        out = '{:1.5G}'.format(val)
        if 'E' in out:
            out = '{}(0)E{}'.format(*out.split('E'))
        else:
            out = out + ' (0)'
        return out
    elif np.log10(np.abs(err)) > 0.:
        # Error > 0
        sigfig = np.ceil(np.log10(np.abs(err))) - 1
        dec = 0.
    elif np.isnan(err):
        # nan error
        return '{} (-)'.format(val)
    else:
        # error < 0
        sigfig = np.floor(np.log10(np.abs(err)) + 0.025)
        dec = -sigfig

    # Round value and error to the number of significant figures
    rval = round(val / (10. ** sigfig)) * (10. ** sigfig)
    rerr = round(err / (10. ** sigfig)) * (10. ** sigfig)
    # size of value and error
    pw = np.floor(np.log10(np.abs(rval)))
    pwr = np.floor(np.log10(np.abs(rerr)))

    max_pw = max(pw, pwr)
    ln = max_pw - sigfig  # power difference

    if np.log10(np.abs(err)) < 0:
        rerr = err / (10. ** sigfig)

    # Small numbers - exponential notation
    if max_pw < -3.:
        rval = rval / (10. ** max_pw)
        fmt = '{' + '0:1.{:1.0f}f'.format(ln) + '}({1:1.0f})E{2:1.0f}'
        return fmt.format(rval, rerr, max_pw)

    # Large numbers - exponential notation
    if max_pw >= 4.:
        rval = rval / (10. ** max_pw)
        rerr = rerr / (10. ** sigfig)
        fmt = '{' + '0:1.{:1.0f}f'.format(ln) + '}({1:1.0f})E+{2:1.0f}'
        return fmt.format(rval, rerr, max_pw)

    fmt = '{' + '0:0.{:1.0f}f'.format(dec + 0) + '} ({1:1.0f})'
    return fmt.format(rval, rerr)


def error_func(y):
    """Default error function"""
    return np.sqrt(np.abs(y) + 1)


def load_xye(filename, delimiter=None):
    """
    Load (x,y,error) from file
      File can have several formats:
        > same as np.savetxt(filename, (x, y, error))
        > x y error\n
        > x, y, error\n
    :param filename: str name of file
    :param delimiter: str alternative delimiter to try
    :return: xdata, ydata, yerror
    """
    try:
        data = np.loadtxt(filename)
    except ValueError:
        if delimiter is None:
            delimiter = ','
        data = np.loadtxt(filename, delimiter=delimiter)
    # Check shape
    if data.shape[1] < data.shape[0]:  # (*, 3/2)
        xdata = data[:, 0]
        ydata = data[:, 1]
        if data.shape[1] < 3:
            error = np.zeros_like(ydata)
        else:
            error = data[:, 2]
    else:
        xdata = data[0, :]
        ydata = data[1, :]
        if data.shape[0] < 3:
            error = np.zeros_like(ydata)
        else:
            error = data[2, :]
    return xdata, ydata, error


def save_results_json(results, filename):
    """
    Save results dict as json file
    :param results: dict
    :param filename: str
    :return: None
    """
    with open(filename, 'w') as fp:
        json.dump(results, fp, indent=4, default=lambda o: '<not serializable>')


def peak_ratio(y, yerror=None):
    """
    Return the ratio signal / error for given dataset
    From Blessing, J. Appl. Cryst. (1997). 30, 421-426 Equ: (1) + (6)
      peak_ratio = (sum((y-bkg)/dy^2)/sum(1/dy^2)) / sqrt(i/sum(1/dy^2))
    :param y: array of y data
    :param yerror: array of errors on data, or None to calcualte np.sqrt(y+0.001)
    :return: float ratio signal / err
    """
    if yerror is None:
        yerror = error_func(y)
    bkg = np.min(y)
    wi = 1 / yerror ** 2
    signal = np.sum(wi * (y - bkg)) / np.sum(wi)
    err = np.sqrt(len(y) / np.sum(wi))
    return signal / err


def gen_weights(yerrors=None):
    """
    Generate weights for fitting routines
    :param yerrors: array(n) or None
    :return: array(n) or None
    """
    if yerrors is None or np.all(np.abs(yerrors) < 0.001):
        weights = None
    else:
        yerrors = np.asarray(yerrors, dtype=float)
        yerrors[yerrors < 1] = 1.0
        weights = 1 / yerrors
        weights = np.abs(np.nan_to_num(weights))
    return weights


def gauss(x, y=None, height=1, cen=0, fwhm=0.5, bkg=0):
    """
    Define Gaussian distribution in 1 or 2 dimensions
    From http://fityk.nieto.pl/model.html
        x = [1xn] array of values, defines size of gaussian in dimension 1
        y = None* or [1xm] array of values, defines size of gaussian in dimension 2
        height = peak height
        cen = peak centre
        fwhm = peak full width at half-max
        bkg = background
    """

    if y is None:
        y = cen

    x = np.asarray(x, dtype=np.float).reshape([-1])
    y = np.asarray(y, dtype=np.float).reshape([-1])
    xx, yy = np.meshgrid(x, y)
    g = height * np.exp(-np.log(2) * (((xx - cen) ** 2 + (yy - cen) ** 2) / (fwhm / 2) ** 2)) + bkg

    if len(y) == 1:
        g = g.reshape([-1])
    return g


def group_adjacent(values, close=10):
    """
    Average adjacent values in array, return grouped array and indexes to return groups to original array
    E.G.
     grp, idx = group_adjacent([1,2,3,10,12,31], close=3)
     grp -> [2, 11, 31]
     idx -> [[0,1,2], [3,4], [5]]

    :param values: array of values to be grouped
    :param close: float
    :return grouped_values: float array(n) of grouped values
    :return indexes: [n] list of lists, each item relates to an averaged group, with indexes from values
    """
    # Check distance between good peaks
    dist_chk = []
    dist_idx = []
    gx = 0
    dist = [values[gx]]
    idx = [gx]
    while gx < len(values) - 1:
        gx += 1
        if (values[gx] - values[gx - 1]) < close:
            dist += [values[gx]]
            idx += [gx]
            # print('Close %2d %2d %2d  %s' % (gx, indexes[gx], indexes[gx-1], dist))
        else:
            dist_chk += [np.mean(dist)]
            dist_idx += [idx]
            dist = [values[gx]]
            idx = [gx]
            # print('Next %2d %2d %2d %s' % (gx, indexes[gx], indexes[gx-1], dist_chk))
    dist_chk += [np.mean(dist)]
    dist_idx += [idx]
    # print('Last %2d %2d %2d %s' % (gx, indexes[gx], indexes[gx-1], dist_chk))
    return np.array(dist_chk), dist_idx