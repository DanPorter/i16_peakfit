"""
i16_peakfit.fitting2d
"""

import numpy as np
import h5py
from imageio import imread
import lmfit
from lmfit.lineshapes import gaussian, lorentzian


def read_tiff(filename):
    """Read TIFF image"""
    if issubclass(type(filename), str):
        return imread(filename)
    return np.array([imread(file) for file in filename])


# From lmfit V1.0+
def gaussian2d(x, y=0.0, amplitude=1.0, centerx=0.0, centery=0.0, sigmax=1.0,
               sigmay=1.0):
    """Return a 2-dimensional Gaussian function.
    gaussian2d(x, y, amplitude, centerx, centery, sigmax, sigmay) =
        amplitude/(2*pi*sigmax*sigmay) * exp(-(x-centerx)**2/(2*sigmax**2)
                                             -(y-centery)**2/(2*sigmay**2))
    """
    z = amplitude*(gaussian(x, amplitude=1, center=centerx, sigma=sigmax) *
                   gaussian(y, amplitude=1, center=centery, sigma=sigmay))
    return z


# From https://lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html#two-dimensional-off-axis-lorentzian
def lorentzian2d(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1., rotation=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    r = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*lorentzian(r)/(np.pi*sigmax*sigmay)


def flat2d(x, y, amplitude=0.):
    """
    Flat 2D background
    :param x:
    :param y:
    :param amplitude:
    :return:
    """
    return amplitude


MODELS_2D = {
    'gaussian': gaussian2d,
    'lorentzian': lorentzian2d,
    'flat': flat2d,
}


MODEL_NAMES_2D = {
    'gaussian': ['gaussian', 'gaussian2d'],
    'lorentzian': ['lorentzian', 'lorentz', 'lorentzian2d'],
    'flat': ['flat', 'flat2d']
}


def get_2dmodel_function(name):
    """
    Get 2D lmfit model function from name
    Name can be quite general and will find it
    :param name: str name of model
    :return: lmfit model
    """
    name = name.lower()
    for model_key, model_names in MODEL_NAMES_2D.items():
        if name in model_names and model_key in MODELS_2D:
            return MODELS_2D[model_key]
    raise KeyError('Not an 2D lmfit method: %s' % name)


def create2dmodel(model_name, prefix='p1_'):
    """
    Generate 2D lmfit model from name
    Name can be quite general and will find it
    :param model_name: str name of model
    :param prefix: model prefix
    :return: lmfit model
    """
    mod = get_2dmodel_function(model_name)
    return lmfit.Model(mod, independent_vars=['x', 'y'], prefix=prefix)


def createGaussian2D(prefix='p1_'):
    """
    Create 2D Gaussian
    :param prefix:
    :return:
    """
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    height_factor = 1. / 2 * np.pi

    fwhmx_expr = f"{fwhm_factor:.7f}*{prefix:s}sigmax"
    fwhmy_expr = f"{fwhm_factor:.7f}*{prefix:s}sigmay"
    height_expr = f"{height_factor:.7f}*{prefix:s}amplitude/(max(1.0e-15, {prefix:s}sigmax)*max(1.0e-15, {prefix:s}sigmay))"

    mod = lmfit.Model(gaussian2d, independent_vars=['x', 'y'], prefix=prefix)
    mod.set_param_hint(f'{prefix:s}sigmax', min=0)
    mod.set_param_hint(f'{prefix:s}sigmay', min=0)
    mod.set_param_hint(f'{prefix:s}fwhmx', expr=fwhmx_expr)
    mod.set_param_hint(f'{prefix:s}fwhmy', expr=fwhmy_expr)
    mod.set_param_hint(f'{prefix:s}height', expr=height_expr)
    return mod

