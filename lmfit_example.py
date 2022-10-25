"""
Example of pure lmfit

Methods: https://lmfit.github.io/lmfit-py/fitting.html#using-the-minimizer-class
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, LinearModel, ExponentialModel


def gauss(xdata, height=1, cen=0, fwhm=0.5, bkg=0):
    """
    Define Gaussian distribution in 1D
    From http://fityk.nieto.pl/model.html
    """
    xdata = np.asarray(xdata, dtype=np.float).reshape([-1])
    g = height * np.exp(-np.log(2) * ((xdata - cen) ** 2 / (fwhm / 2) ** 2)) + bkg
    return g


# Create peaks
x = np.arange(-10, 10, 0.1)
y1 = gauss(x, height=50, cen=3, fwhm=0.5, bkg=1)  # single peak
w1 = 1 / np.sqrt(y1)
y2 = y1 + gauss(x, height=60, cen=1, fwhm=0.7, bkg=0)  # double peak
w2 = 1/ np.sqrt(y2+1)

# ------------ Fit 1 Peak ---------------------
# Define lmfit models
peak_mod = PseudoVoigtModel()
bkg_mod = LinearModel()


# Guess initial parameters
pars = peak_mod.guess(y1, x=x)
pars += bkg_mod.make_params(intercept=np.min(y1), slope=0)

# Fit
mod = peak_mod + bkg_mod
res = mod.fit(y1, pars, x=x, weights=w1)

print(res.fit_report())
res.plot()
plt.show()


#------------ Fit 2 Peaks ---------------------
# Define lmfit models
peak_mod1 = PseudoVoigtModel(prefix='p1_')
peak_mod2 = PseudoVoigtModel(prefix='p2_')
bkg_mod = LinearModel(prefix='bkg_')


# Guess initial parameters
#pars = peak_mod1.guess(y2, x=x)
#pars += peak_mod2.guess(y2, x=x)
#pars += bkg_mod.make_params(intercept=np.min(y2), slope=0)

mod = peak_mod1 + peak_mod2 + bkg_mod
pars = mod.make_params()
pars['p1_amplitude'].set(value=np.sum(y2 - y2.min()) / 2)
pars['p2_amplitude'].set(value=np.sum(y2 - y2.min()) / 2)
pars['p1_sigma'].set(value=np.mean(3*np.diff(x)))
pars['p2_sigma'].set(value=np.mean(3*np.diff(x)))
pars['p1_center'].set(value=np.percentile(x, 25))
pars['p2_center'].set(value=np.percentile(x, 75))
pars['bkg_slope'].set(value=0)
pars['bkg_intercept'].set(value=np.min(y2))

# Fit
init = mod.eval(pars, x=x)
res = mod.fit(y2, pars, x=x, weights=w2, method='ampgo')

#print(res.fit_report())
#res.plot()
#plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=60)
fig.suptitle('Fit 2 models')
axes[0].plot(x, y2, 'o', label='data')
axes[0].plot(x, init, '--', label='initial fit')
axes[0].plot(x, res.best_fit, '-', label='best fit')
axes[0].legend()

comps = res.eval_components(x=x)
axes[1].plot(x, y2, 'o', label='Data')
axes[1].plot(x, comps['p1_'], '--', label='Component 1')
axes[1].plot(x, comps['p2_'], '--', label='Component 2')
axes[1].plot(x, comps['bkg_'], '--', label='Background')
axes[1].legend()

plt.show()

