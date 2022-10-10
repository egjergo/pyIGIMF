import numpy as np
from scipy import optimize
import scipy.integrate as integr
from mpmath import mp
    
def weighted_func(M, func, *args, **kwargs):
    return np.multiply(M, func(M, *args, **kwargs))

def power_law(alpha):
    return lambda m: m**(-alpha)

def get_norm(IMF, Mtot, min_val, max_val, *args, **kwargs):
    '''duplicate of stellar_IMF !!!!!!! '''
    k, M = normalization(IMF, Mtot, min_val, min_val, *args, **kwargs)
    IMF_func = lambda mass: k * IMF(mass, m_max=M, *args, **kwargs)
    IMF_weighted_func = lambda mass: weighted_func(mass, IMF_func, *args, **kwargs)
    return k, M, np.vectorize(IMF_func), np.vectorize(IMF_weighted_func)

def normalized(x, func, condition=None, *args, **kwargs):
    ''' IMF behavior depending on whether or not it has been normalized '''
    if condition:
        if x <= condition:
            return func
        else:
            return 0.
    else:
        return func
        
def normalization(IMF, Mtot, lower_lim, upper_lim, *args, **kwargs) -> (float, float):
    r'''
    Function that extracts k and m_max (blue boxes in the notes)
    IMF:    mass distribution function, i.e. either Eq. (1) or (8)
    Mtot:   value of the integrals in Eq. (2) and (9)
    k:      normalization of the IMF function, i.e. Eq. (3) and (10)
    upper_lim and lower_lim:    
            global upper and lower limits on the integrals
    guess:  guess on the local upper (lower) limit on the integrals 
            of Eq.2 and Eq.9 (of Eq.3 and Eq.10)
    x:      evaluated local upper (lower) limit on the integrals
            of Eq.2 and Eq.9 (of Eq.3 and Eq.10)
    *args   other required arguments
    **kwargs    optional keyword arguments
        
    -----
        
    .. math::
    `M = \int_{\mathrm{lower_lim}}^{\mathrm{m_max}}
    m \, \mathrm{IMF}(m,...)\,\mathrm{d}m`
        
    .. math::
    `1 = \int_{\mathrm{m_max}}^{{\rm upper_lim}}{\mathrm{IMF}(m,...)} \,\mathrm{d}m`
    '''
    k = lambda x: np.reciprocal(integr.quad(IMF, x, upper_lim, args=(args))[0]) # quad sometimes gives negative k
    def weighted_IMF(m, x, *args):
        return m * IMF(m, *args) * k(x)
    func = lambda x: (integr.quad(weighted_IMF, lower_lim, x, args=(x, *args))[0] - Mtot)
    sol = optimize.root_scalar(func, bracket=[lower_lim, upper_lim], rtol=1e-8)
    m_max = sol.root
    return k(m_max), m_max


def normalization_ECMF(IMF, beta, Mtot, lower_lim, upper_lim, *args) -> (float, float):
    k = lambda x: np.divide(1-beta, upper_lim**(1-beta) - x**(1-beta))
    def weighted_IMF(m, x, *args):
        return m * IMF(m, *args) * k(x)
    func = lambda x: (integr.quad(weighted_IMF, lower_lim, x, args=(x, *args))[0] - Mtot)
    sol = optimize.root_scalar(func, x0=1e0, x1=1e5, rtol=1e-8)
    m_max = sol.root
    return k(m_max), m_max



def normalization_IMF(alpha1, alpha2, alpha3, Mtot, lower_lim, upper_lim) -> (float, float):
    def integral_IMF(ll, ul, power):
        if ll < ul:
            return np.divide(ul**(1-power) - ll**(1-power), 1-power)
        else:
            return 0.
    def k(x):
        if np.logical_and(x >= lower_lim, x < 0.5):
            return (np.reciprocal(2 * integral_IMF(x, 0.5, alpha1) 
                                + integral_IMF(0.5, 1., alpha2) 
                                + integral_IMF(1., upper_lim, alpha3)))
        if np.logical_and(x >= 0.5, x < 1.):
            return (np.reciprocal(integral_IMF(x, 1., alpha2) 
                                + integral_IMF(1., upper_lim, alpha3)))
        if np.logical_and(x >= 1., x <= upper_lim):
            return (np.reciprocal(integral_IMF(x, upper_lim, alpha3)))
        else:
            return 0.
    def weighted_IMF(x):
        if np.logical_and(x >= lower_lim, x < 0.5):
            return (2 * integral_IMF(0.08, x, alpha1-1))
        if np.logical_and(x >= 0.5, x < 1.):
            return (2 * integral_IMF(0.08, 0.5, alpha1-1)
                    + integral_IMF(0.5, x, alpha2-1))
        if np.logical_and(x >= 1., x <= upper_lim):
            return (2 * integral_IMF(0.08, 0.5, alpha1-1)
                    + integral_IMF(0.5, 1., alpha2-1)
                    + integral_IMF(1., x, alpha3-1))
        else:
            return 0.
    #def integral_weighted_IMF(m, x, alpha3, ll, ul, power):
    func = lambda x: (k(x) * weighted_IMF(x) - Mtot)
    #sol = optimize.root_scalar(func, x0=1, x1=20, rtol=1e-8)
    try:
        sol = optimize.root_scalar(func, method='bisect', rtol=1e-15, bracket=(lower_lim, upper_lim))
        m_max = sol.root
    except:
        m_max = upper_lim
    #print(f'{Mtot = },\t {m_max = },\t {k(m_max)=}')
    return k(m_max), m_max