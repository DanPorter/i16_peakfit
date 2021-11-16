"""
Fitting functions using lmfit

See: https://lmfit.github.io/lmfit-py/builtin_models.html

Use of peakfit:
from fitting import peakfit
fit = peakfit(xdata, ydata)  # returns lmfit object
print(fit)
fit.plot()
"""

import numpy as np
#from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, LinearModel, ExponentialModel
from lmfit import models

# https://lmfit.github.io/lmfit-py/builtin_models.html#peak-like-models
MODELS = {
    'gaussian': models.GaussianModel,
    'lorentzian': models.LorentzianModel,
    'voigt': models.VoigtModel,
    'pseudovoigt': models.PseudoVoigtModel,
    'linear': models.LinearModel,
    'exponential': models.ExponentialModel,
    'breitwigner': models.BreitWignerModel,
    'complexconstant': models.ComplexConstantModel,
    'constant': models.ConstantModel,
    'dampedharmonicoscillator': models.DampedHarmonicOscillatorModel,
    'dampedoscillator': models.DampedOscillatorModel,
    'donaich': models.DonaichModel,
    'exponentialgaussian': models.ExponentialGaussianModel,
    'lognormal': models.LognormalModel,
    'moffat': models.MoffatModel,
    'parabolic': models.ParabolicModel,
    'pearson7': models.Pearson7Model,
    'polynomial': models.PolynomialModel,
    'powerlaw': models.PowerLawModel,
    'quadratic': models.QuadraticModel,
    'rectangle': models.RectangleModel,
    'skewedgaussian': models.SkewedGaussianModel,
    'skewedvoigt': models.SkewedVoigtModel,
    'splitlorentzian': models.SplitLorentzianModel,
    'step': models.StepModel,
    'studentst': models.StudentsTModel,
}

PEAK_MODELS = {
    'gaussian': ['gaussian', 'gauss'],
    'voigt': ['voight', 'voigt'],
    'pseudovoigt': ['pseudovoight', 'pvoight', 'pseudovoigt', 'pvoigt'],
    'lorentzian': ['lorentz', 'lorentzian', 'lor'],
    'splitlorentzian': ['splitlorentzian', 'split_lorentzian', 'splitlorentz', 'skewedlorentzian'],
    'moffat': ['moffat'],
    'pearson7': ['pearson7', 'pearson'],
    'studentst': ['studentst', 'students_t', 'students', 'tmodel'],
    'breitwigner': ['breitwigner', 'breit_wigner', 'breit-wigner-fano', 'brietwigner', 'breit', 'wigner'],
    'lognormal': ['lognormal'],
    'dampedoscillator': ['dampedoscillator', 'damped_oscillator'],
    'dampedharmonicoscillator': ['dampedharmonicoscillator', 'dho'],
    'exponentialgaussian': ['exponentialgaussian', 'expgaussian'],
    'skewedgaussian': ['skewedgaussian', 'skewed_gaussian', 'splitgaussian'],
    'skewedvoigt': ['skewedvoigt', 'skewed_voigt', 'splitvoigt', 'skewedvoight', 'splitvoight'],
    'donaich': ['donaich'],
}

BACKGROUND_MODELS = {
    'linear': ['flat', 'slope', 'linear', 'line', 'straight'],
    'exponential': ['exponential', 'curve'],
    'constant': ['constant'],
    'quadratic': ['quadratic'],
    'polynomial': ['polynomial'],
    'sine': ['sine', 'sin', 'cos', 'cosine'],
    'step': ['step'],
    'rectangle': ['rectangle'],
    'powerlaw': ['powerlaw']
}

# https://lmfit.github.io/lmfit-py/fitting.html#fit-methods-table
METHODS = {
    'leastsq': 'Levenberg-Marquardt',
    'nelder': 'Nelder-Mead',
    'lbfgsb': 'L-BFGS-B',
    'powell': 'Powell',
    'cg': 'Conjugate Gradient',
    'newton': 'Newton-CG',
    'cobyla': 'COBYLA',
    'bfgsb': 'BFGS',
    'tnc': 'Truncated Newton',
    'trust-ncg': 'Newton CG trust-region',
    'trust-exact': 'Exact trust-region',
    'trust-krylov': 'Newton GLTR trust-region',
    'trust-constr': 'Constrained trust-region',
    'dogleg': 'Dogleg',
    'slsqp': 'Sequential Linear Squares Programming',
    'differential_evolution': 'Differential Evolution',
    'brute': 'Brute force method',
    'basinhopping': 'Basinhopping',
    'ampgo': 'Adaptive Memory Programming for Global Optimization',
    'shgo': 'Simplicial Homology Global Ooptimization',
    'dual_annealing': 'Dual Annealing',
    'emcee': 'Maximum likelihood via Monte-Carlo Markov Chain',
}


def getmodel(name):
    """
    Return lmfit model from name
    Name can be quite general and will find it
    :param name: str name of model
    :return: lmfit model
    """
    name = name.lower()
    if name in MODELS:
        return MODELS[name]
    for model_key, model_names in PEAK_MODELS.items():
        if name in model_names:
            return MODELS[model_key]
    for model_key, model_names in BACKGROUND_MODELS.items():
        if name in model_names:
            return MODELS[model_key]
    raise KeyError('Not an lmfit method: %s' % name)


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


def generate_model(xvals, yvals, yerrors=None,
                   npeaks=None, min_peak_power=None, points_between_peaks=6,
                   model='Gaussian', background='slope', initial_parameters=None, fix_parameters=None):
    """
    Generate lmfit profile models
    See: https://lmfit.github.io/lmfit-py/builtin_models.html#example-3-fitting-multiple-peaks-and-using-prefixes
    E.G.:
      mod, pars = generate_model(x, y, npeaks=1, model='Gauss', backgroud='slope')

    Peak Search:
     The number of peaks and initial peak centers will be estimated using the find_peaks function. If npeaks is given,
     the largest npeaks will be used initially. 'min_peak_power' and 'peak_distance_idx' can be input to tailor the
     peak search results.
     If the peak search returns < npeaks, fitting parameters will initially choose npeaks equally distributed points

    Peak Models:
     Choice of peak model: 'Gaussian', 'Lorentzian', 'Voight',' PseudoVoight'
    Background Models:
     Choice of background model: 'slope', 'exponential'

    :param xvals: array(n) position data
    :param yvals: array(n) intensity data
    :param yerrors: None or array(n) - error data to pass to fitting function as weights: 1/errors^2
    :param npeaks: None or int number of peaks to fit. None will guess the number of peaks
    :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
    :param points_between_peaks: int, group adjacent maxima if closer in index than this
    :param model: str or lmfit.Model, specify the peak model 'Gaussian','Lorentzian','Voight'
    :param background: str, specify the background model: 'slope', 'exponential'
    :param initial_parameters: None or dict of initial values for parameters
    :param fix_parameters: None or dict of parameters to fix at positions
    :return: lmfit.model.ModelResult < fit results object
    """
    xvals = np.asarray(xvals, dtype=float).reshape(-1)
    yvals = np.asarray(yvals, dtype=float).reshape(-1)

    # Find peaks
    peak_idx, peak_pow = find_peaks(yvals, yerrors, min_peak_power, points_between_peaks)
    peak_centers = {'p%d_center' % (n + 1): xvals[peak_idx[n]] for n in range(len(peak_idx))}
    if npeaks is None:
        npeaks = len(peak_centers)

    if initial_parameters is None:
        initial_parameters = {}
    if fix_parameters is None:
        fix_parameters = {}

    peak_mod = None
    bkg_mod = None
    for model_name, names in PEAK_MODELS.items():
        if model.lower() in names:
            peak_mod = MODELS[model_name]
    for model_name, names in BACKGROUND_MODELS.items():
        if background.lower() in names:
            bkg_mod = MODELS[model_name]

    mod = bkg_mod(prefix='bkg_')
    for n in range(npeaks):
        mod += peak_mod(prefix='p%d_' % (n + 1))

    pars = mod.make_params()

    # initial parameters
    min_wid = np.mean(np.diff(xvals))
    max_wid = xvals.max() - xvals.min()
    area = (yvals.max() - yvals.min()) * (3 * min_wid)
    percentile = np.linspace(0, 100, npeaks + 2)
    for n in range(1, npeaks + 1):
        pars['p%d_amplitude' % n].set(value=area / npeaks, min=0)
        pars['p%d_sigma' % n].set(value=3 * min_wid, min=min_wid, max=max_wid)
        pars['p%d_center' % n].set(value=np.percentile(xvals, percentile[n]), min=xvals.min(), max=xvals.max())
    # find_peak centers
    for ipar, ival in peak_centers.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=True)
    # user input parameters
    for ipar, ival in initial_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=True)
    for ipar, ival in fix_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=False)
    return mod, pars


def generate_model_script(xvals, yvals, yerrors=None,
                          npeaks=None, min_peak_power=None, points_between_peaks=6,
                          model='Gaussian', background='slope', initial_parameters=None, fix_parameters=None,
                          include_i16_peakfit=True):
    """
    Generate script to create lmfit profile models
    E.G.:
      string = generate_mode_stringl(x, y, npeaks=1, model='Gauss', backgroud='slope')

    :param xvals: array(n) position data
    :param yvals: array(n) intensity data
    :param yerrors: None or array(n) - error data to pass to fitting function as weights: 1/errors^2
    :param npeaks: None or int number of peaks to fit. None will guess the number of peaks
    :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
    :param points_between_peaks: int, group adjacent maxima if closer in index than this
    :param model: str or lmfit.Model, specify the peak model 'Gaussian','Lorentzian','Voight'
    :param background: str, specify the background model: 'slope', 'exponential'
    :param initial_parameters: None or dict of initial values for parameters
    :param fix_parameters: None or dict of parameters to fix at positions
    :param include_i16_peakfit: if False, only include lmfit imports
    :return: str
    """

    data = "xdata = np.array(%s)\n" % list(xvals)
    data += "ydata = np.array(%s)\n" % list(yvals)
    if yerrors is None or np.all(np.abs(yerrors) < 0.001):
        data += 'yerrors = None\n'
        data += 'weights = None\n\n'
    else:
        data += "yerrors = np.array(%s)\n" % list(yerrors)
        data += "yerrors[yerrors < 1] = 1.0\n"
        data += "weights = 1 / yerrors\n\n"

    if initial_parameters is None:
        initial_parameters = {}
    if fix_parameters is None:
        fix_parameters = {}
    params = "initial = %s\nfixed = %s\n" % (initial_parameters, fix_parameters)

    if include_i16_peakfit:
        out = "import numpy as np\nfrom i16_peakfit import fitting\n\n"
        out += data
        out += '%s\n' % params
        out += "mod, pars = fitting.generate_model(xdata, ydata, yerrors,\n" \
               "                                   npeaks=%s, min_peak_power=%s, points_between_peaks=%s,\n" \
               "                                   model='%s', background='%s',\n" \
               "                                   initial_parameters=initial, fix_parameters=fixed)\n" % (
                   npeaks, min_peak_power, points_between_peaks, model, background
               )
    else:
        # Find peaks
        peak_idx, peak_pow = find_peaks(yvals, yerrors, min_peak_power, points_between_peaks)
        peak_centers = {'p%d_center' % (n + 1): xvals[peak_idx[n]] for n in range(len(peak_idx))}
        for model_name, names in PEAK_MODELS.items():
            if model.lower() in names:
                peak_mod = MODELS[model_name]
        for model_name, names in BACKGROUND_MODELS.items():
            if background.lower() in names:
                bkg_mod = MODELS[model_name]
        peak_name = peak_mod.__name__
        bkg_name = bkg_mod.__name__

        out = "import numpy as np\nfrom lmfit import models\n\n"
        out += data
        out += "%speak_centers = %s\n\n" % (params, peak_centers)
        out += "mod = models.%s(prefix='bkg_')\n" % bkg_name
        out += "for n in range(len(peak_centers)):\n    mod += models.%s(prefix='p%%d_' %% (n+1))\n" % peak_name
        out += "pars = mod.make_params()\n\n"
        out += "# initial parameters\n"
        out += "min_wid = np.mean(np.diff(xdata))\n"
        out += "max_wid = xdata.max() - xdata.min()\n"
        out += "area = (ydata.max() - ydata.min()) * (3 * min_wid)\n"
        out += "for n in range(1, len(peak_centers)+1):\n"
        out += "    pars['p%d_amplitude' % n].set(value=area/len(peak_centers), min=0)\n"
        out += "    pars['p%d_sigma' % n].set(value=3*min_wid, min=min_wid, max=max_wid)\n"
        out += "# find_peak centers\n"
        out += "for ipar, ival in peak_centers.items():\n"
        out += "    if ipar in pars:\n"
        out += "        pars[ipar].set(value=ival, vary=True)\n"
        out += "# user input parameters\n"
        out += "for ipar, ival in initial.items():\n"
        out += "    if ipar in pars:\n"
        out += "        pars[ipar].set(value=ival, vary=True)\n"
        out += "for ipar, ival in fixed.items():\n"
        out += "    if ipar in pars:\n"
        out += "        pars[ipar].set(value=ival, vary=False)\n\n"
    out += "# Fit data\n"
    out += "res = mod.fit(ydata, pars, x=xdata, weights=weights, method='leastsqr')\n"
    out += "print(res.fit_report())\n\n"
    out += "fig, grid = res.plot()\n"
    out += "ax1, ax2 = fig.axes\n"
    out += "comps = res.eval_components()\n"
    out += "for component in comps.keys():\n"
    out += "    ax2.plot(xdata, comps[component], label=component)\n"
    out += "    ax2.legend()\n"
    out += "fig.show()\n\n"
    return out


def peak_results(res):
    """
    Generate totals dict
    :param res: lmfit_result
    :return: {totals: (value, error)}
    """
    peak_prefx = [mod.prefix for mod in res.components if 'bkg' not in mod.prefix]
    npeaks = len(peak_prefx)
    nn = 1 / len(peak_prefx) if len(peak_prefx) > 0 else 1
    comps = res.eval_components()
    fit_dict = {
        'lmfit': res,
        'npeaks': npeaks,
        'chisqr': res.chisqr,
        'xdata': res.userkws['x'],
        'ydata': res.data,
        'weights': res.weights,
        'yerror': 1 / res.weights if res.weights is not None else 0 * res.data,
        'yfit': res.best_fit,
    }
    for comp_prefx, comp in comps.items():
        fit_dict['%sfit' % comp_prefx] = comp
    for pname, param in res.params.items():
        ename = 'stderr_' + pname
        fit_dict[pname] = param.value
        fit_dict[ename] = param.stderr if param.stderr is not None else 0
    totals = {
        'amplitude': np.sum([res.params['%samplitude' % pfx].value for pfx in peak_prefx]),
        'center': np.mean([res.params['%scenter' % pfx].value for pfx in peak_prefx]),
        'sigma': np.mean([res.params['%ssigma' % pfx].value for pfx in peak_prefx]),
        'height': np.mean([res.params['%sheight' % pfx].value for pfx in peak_prefx]),
        'fwhm': np.mean([res.params['%sfwhm' % pfx].value for pfx in peak_prefx]),
        'background': np.mean(comps['bkg_']),
        'stderr_amplitude': np.sqrt(np.sum([fit_dict['stderr_%samplitude' % pfx] ** 2 for pfx in peak_prefx])),
        'stderr_center': np.sqrt(np.sum([fit_dict['stderr_%scenter' % pfx] ** 2 for pfx in peak_prefx])) * nn,
        'stderr_sigma': np.sqrt(np.sum([fit_dict['stderr_%ssigma' % pfx] ** 2 for pfx in peak_prefx])) * nn,
        'stderr_height': np.sqrt(np.sum([fit_dict['stderr_%sheight' % pfx] ** 2 for pfx in peak_prefx])) * nn,
        'stderr_fwhm': np.sqrt(np.sum([fit_dict['stderr_%sfwhm' % pfx] ** 2 for pfx in peak_prefx])) * nn,
    }
    fit_dict.update(totals)
    return fit_dict


def peak_results_str(res):
    """
    Generate output str from lmfit results, including totals
    :param res: lmfit_result
    :return: str
    """
    fit_dict = peak_results(res)
    out = 'Fit Results\n'
    out += '%s\n' % res.model.name
    out += 'Npeaks = %d\n' % fit_dict['npeaks']
    out += 'Method: %s => %s\n' % (res.method, res.message)
    out += 'Chisqr = %1.5g\n' % res.chisqr
    # Peaks
    peak_prefx = [mod.prefix for mod in res.components if 'bkg' not in mod.prefix]
    for prefx in peak_prefx:
        out += '\nPeak %s\n' % prefx
        for pn in res.params:
            if prefx in pn:
                out += '%15s = %s\n' % (pn, stfm(fit_dict[pn], fit_dict['stderr_%s' % pn]))

    out += '\nBackground\n'
    for pn in res.params:
        if 'bkg' in pn:
            out += '%15s = %s\n' % (pn, stfm(fit_dict[pn], fit_dict['stderr_%s' % pn]))

    out += '\nTotals\n'
    out += '      amplitude = %s\n' % stfm(fit_dict['amplitude'], fit_dict['stderr_amplitude'])
    out += '         center = %s\n' % stfm(fit_dict['center'], fit_dict['stderr_center'])
    out += '         height = %s\n' % stfm(fit_dict['height'], fit_dict['stderr_height'])
    out += '          sigma = %s\n' % stfm(fit_dict['sigma'], fit_dict['stderr_sigma'])
    out += '           fwhm = %s\n' % stfm(fit_dict['fwhm'], fit_dict['stderr_fwhm'])
    out += '     background = %s\n' % stfm(fit_dict['background'], 0)
    return out


def peak_results_fit(res, ntimes=10):
    """
    Generate xfit, yfit data, interpolated to give smoother variation
    :param res: lmfit_result
    :param ntimes: int, number of points * old number of points
    :return: xfit, yfit
    """
    old_x = res.userkws['x']
    xfit = np.linspace(np.min(old_x), np.max(old_x), np.size(old_x) * ntimes)
    yfit = res.eval(x=xfit)
    return xfit, yfit


def peak_results_plot(res, axes=None, xlabel=None, ylabel=None, title=None):
    """
    Plot peak results
    :param res: lmfit result
    :param axes: None or matplotlib axes
    :param xlabel: None or str
    :param ylabel: None or str
    :param title: None or str
    :return: matplotlib figure or axes
    """
    xdata = res.userkws['x']
    if title is None:
        title = res.model.name

    if axes:
        ax = res.plot_fit(ax=axes, xlabel=xlabel, ylabel=ylabel)
        # Add peak components
        comps = res.eval_components(x=xdata)
        for component in comps.keys():
            ax.plot(xdata, comps[component], label=component)
            ax.legend()
        return ax

    fig, grid = res.plot(xlabel=xlabel, ylabel=ylabel)
    ax1, ax2 = fig.axes
    ax1.set_title(title, wrap=True)
    # Add peak components
    comps = res.eval_components(x=xdata)
    for component in comps.keys():
        ax2.plot(xdata, comps[component], label=component)
        ax2.legend()
    fig.show()
    return fig


def modelfit(xvals, yvals, yerrors=None, model=None, initial_parameters=None, fix_parameters=None,
             method='leastsq', print_result=False, plot_result=False):
    """
    Fit x,y data to a model from lmfit
    E.G.:
      res = peakfit(x, y, model='Gauss')
      print(res.fit_report())
      res.plot()
      val = res.params['amplitude'].value
      err = res.params['amplitude'].stderr

    Model:
     from lmfit import models
     model1 = model.GaussianModel()
     model2 = model.LinearModel()
     model = model1 + model2
     res = model.fit(y, x=x)

    Provide initial guess:
      res = modelfit(x, y, model=VoightModel(), initial_parameters={'center':1.23})

    Fix parameter:
      res = modelfit(x, y, model=VoightModel(), fix_parameters={'sigma': fwhm/2.3548200})

    :param xvals: array(n) position data
    :param yvals: array(n) intensity data
    :param yerrors: None or array(n) - error data to pass to fitting function as weights: 1/errors^2
    :param model: lmfit.Model
    :param initial_parameters: None or dict of initial values for parameters
    :param fix_parameters: None or dict of parameters to fix at positions
    :param method: str method name, from lmfit fitting methods
    :param print_result: if True, prints the fit results using fit.fit_report()
    :param plot_result: if True, plots the results using fit.plot()
    :return: lmfit.model.ModelResult < fit results object
    """

    xvals = np.asarray(xvals, dtype=float).reshape(-1)
    yvals = np.asarray(yvals, dtype=float).reshape(-1)
    weights = gen_weights(yerrors)

    if initial_parameters is None:
        initial_parameters = {}
    if fix_parameters is None:
        fix_parameters = {}

    if model is None:
        model = models.GaussianModel() + models.LinearModel()

    pars = model.make_params()

    # user input parameters
    for ipar, ival in initial_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=True)
    for ipar, ival in fix_parameters.items():
        if ipar in pars:
            pars[ipar].set(value=ival, vary=False)

    res = model.fit(yvals, pars, x=xvals, weights=weights, method=method)

    if print_result:
        print(res.fit_report())
    if plot_result:
        res.plot()
    return res


def multipeakfit(xvals, yvals, yerrors=None,
                 npeaks=None, min_peak_power=None, points_between_peaks=6,
                 model='Gaussian', background='slope', initial_parameters=None, fix_parameters=None, method='leastsq',
                 print_result=False, plot_result=False):
    """
    Fit x,y data to a model with multiple peaks using lmfit
    See: https://lmfit.github.io/lmfit-py/builtin_models.html#example-3-fitting-multiple-peaks-and-using-prefixes
    E.G.:
      res = multipeakfit(x, y, npeaks=None, model='Gauss', plot_result=True)
      val = res.params['p1_amplitude'].value
      err = res.params['p1_amplitude'].stderr

    Peak Search:
     The number of peaks and initial peak centers will be estimated using the find_peaks function. If npeaks is given,
     the largest npeaks will be used initially. 'min_peak_power' and 'peak_distance_idx' can be input to tailor the
     peak search results.
     If the peak search returns < npeaks, fitting parameters will initially choose npeaks equally distributed points

    Peak Models:
     Choice of peak model: 'Gaussian', 'Lorentzian', 'Voight',' PseudoVoight'
    Background Models:
     Choice of background model: 'slope', 'exponential'

    Peak Parameters (%d=number of peak):
    Parameters in '.._parameters' dicts and in output results. Each peak (upto npeaks) has a set number of parameters:
     'p%d_amplitude', 'p%d_center', 'p%d_dsigma', pvoight only: 'p%d_fraction'
     output only: 'p%d_fwhm', 'p%d_height'
    Background parameters:
     'bkg_slope', 'bkg_intercept', or for exponential: 'bkg_amplitude', 'bkg_decay'

    Provide initial guess:
      res = multipeakfit(x, y, model='Voight', initial_parameters={'p1_center':1.23})

    Fix parameter:
      res = multipeakfit(x, y, model='gauss', fix_parameters={'p1_sigma': fwhm/2.3548200})

    :param xvals: array(n) position data
    :param yvals: array(n) intensity data
    :param yerrors: None or array(n) - error data to pass to fitting function as weights: 1/errors^2
    :param npeaks: None or int number of peaks to fit. None will guess the number of peaks
    :param min_peak_power: float, only return peaks with power greater than this. If None compare against std(y)
    :param points_between_peaks: int, group adjacent maxima if closer in index than this
    :param model: str or lmfit.Model, specify the peak model 'Gaussian','Lorentzian','Voight'
    :param background: str, specify the background model: 'slope', 'exponential'
    :param initial_parameters: None or dict of initial values for parameters
    :param fix_parameters: None or dict of parameters to fix at positions
    :param method: str method name, from lmfit fitting methods
    :param print_result: if True, prints the fit results using fit.fit_report()
    :param plot_result: if True, plots the results using fit.plot()
    :return: lmfit.model.ModelResult < fit results object
    """
    xvals = np.asarray(xvals, dtype=float).reshape(-1)
    yvals = np.asarray(yvals, dtype=float).reshape(-1)
    weights = gen_weights(yerrors)

    mod, pars = generate_model(xvals, yvals, yerrors,
                               npeaks=npeaks, min_peak_power=min_peak_power, points_between_peaks=points_between_peaks,
                               model=model, background=background,
                               initial_parameters=initial_parameters, fix_parameters=fix_parameters)

    # Fit data against model using choosen method
    res = mod.fit(yvals, pars, x=xvals, weights=weights, method=method)

    if print_result:
        print(res.fit_report())
    if plot_result:
        fig, grid = res.plot()
        ax1, ax2 = fig.axes
        # Add peak components
        comps = res.eval_components(x=xvals)
        for component in comps.keys():
            ax2.plot(xvals, comps[component], label=component)
            ax2.legend()
    return res
