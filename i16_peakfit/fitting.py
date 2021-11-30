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
from lmfit import models

# https://lmfit.github.io/lmfit-py/builtin_models.html#peak-like-models
from i16_peakfit.peak_finding import find_peaks
from i16_peakfit.functions import stfm, error_func, gen_weights, group_adjacent

MODELS = {'%s' % mod[:-5].lower(): getattr(models, mod) for mod in dir(models) if mod.endswith('Model')}

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
    'doniach': ['donaich', 'doniach'],
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
        if name in model_names and model_key in MODELS:
            return MODELS[model_key]
    for model_key, model_names in BACKGROUND_MODELS.items():
        if name in model_names and model_key in MODELS:
            return MODELS[model_key]
    raise KeyError('Not an lmfit method: %s' % name)


def new_peak_prefix(model, fmt='p%d_'):
    """
    Determine new peak prefix for model
    :param model: lmfit model
    :param fmt: prefix format
    :return: str : fmt % n where n is peak number
    """
    if model is None or not hasattr(model, 'components'):
        return fmt % 1

    old_prefix = [comp.prefix for comp in model.components]
    n = 1
    while fmt % n in old_prefix:
        n += 1
    return fmt % n


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

    peak_mod = getmodel(model)
    bkg_mod = getmodel(background)

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
        peak_mod = getmodel(model)
        bkg_mod = getmodel(background)
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
        'background': np.mean(comps['bkg_']) if 'bkg_' in comps else 0.0,
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
