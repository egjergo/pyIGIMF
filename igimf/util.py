import numpy as np
from scipy import optimize
import scipy.integrate as integr

def _impose_limit(func, ll, ul, *args, **kwargs):
    if ll < ul:
        return func(*args, **kwargs)
    else:
        return 0.

def _check_limit(func, ll, ul=None, *args, **kwargs):
    '''Imposes upper limit to the mass function when necessary'''
    if ll < ul:
        _impose_limit(func, ll, ul, *args, **kwargs)
    else:
        return func(*args, **kwargs)

def _weighted_func(func, M, *args, **kwargs):
    return np.multiply(M, func(M, *args, **kwargs))

def power_law(k: float, alpha: float):
    '''
    Calculates a decaying power law function for a given m array. 
    
        k: constant of proportionality.
        alpha: exponent or power of m, a positive number.
    '''
    func = lambda m: k * np.power(m, -alpha)
    return np.vectorize(func)

def integrate_power_law(ll: float, ul: float, alpha: float):
    '''
    integral of the (not normalized) decaying power law
    
        ll: lower integration limit
        ul: upper integration limit
        alpha: exponent or power of m, a positive number.
    '''
    power = alpha
    return (ul**power - ll**power) / power

def normalization_ECMF(beta, Mtot, lower_lim, upper_lim, *args) -> (float, float):
    k_ECMF = lambda x: np.reciprocal(integrate_power_law(x, upper_lim, beta))
    weighted_ECMF = lambda x: _impose_limit(integrate_power_law, lower_lim, x, # repeated entry because the
                                            lower_lim, x, beta+1) # second time they enter integrate_power_law
    func = lambda x: (k_ECMF(x) * weighted_ECMF(x) - Mtot)
    sol = optimize.root_scalar(func, x0=1e0, x1=1e5, rtol=1e-15)
    Mecl_max = sol.root
    return k_ECMF(Mecl_max), Mecl_max