import latexify
import numpy as np
from scipy import optimize
import scipy.integrate as integr

def zeros_func(mmin=5, mmax=1e9, beta=2.3):
    return lambda x: (1-beta)/(2-beta) * np.divide(x**(2-beta) - mmin**(2-beta), mmax**(1-beta) - x**(1-beta))

def minimize(func=zeros_func, total=2e7, *args, **kwargs):
    return lambda x: total - func(x, *args, **kwargs)



def mass_weighted_func(M, func, *args, **kwargs):
    '''Mass-weighted function'''
    return np.multiply(M, func(M, *args, **kwargs))

def plaw(M, neg_exponent=2.3):
    '''Decaying power law'''
    #if M > 0.:
    return M**(-neg_exponent)
    #else:
    #    print('Please provide positive masses')

def int_plaw(ll, ul, power):
    r'''
    Integration of the decaying power law
    
    $\int^{u}_{l}x^{\alpha} {\rm d}x = \frac{u^{1-\alpha} - l^{1-\alpha}}{1-\alpha}$
    '''
    if np.logical_or(power==1, power==2):
        return np.log(ul) - np.log(ll)
    else:
        return np.divide(plaw(ul, power-1) - plaw(ll, power-1), 1-power)

def broken_plaw_norm(limit, lower_exp, upper_exp):
    '''Ensure continuity in the broken power law'''
    return limit**(upper_exp - lower_exp)
    
def Kroupa01(Mstar, alpha0=0.3, alpha1=1.3, alpha2=2.3, alpha3=2.3,
            lim01=0.08, lim12=0.5, lim23=1., Ml=0.07, Mu=150):
    '''
    Broken power law. The default parameters are equivalent to 
    Kroupa (2001)
    https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract 
    
    lim refer to the mass limits that break the power law
    '''
    a0 = 1.
    a1 = a0 * broken_plaw_norm(lim01, alpha0, alpha1)
    a2 = a1 * broken_plaw_norm(lim12, alpha1, alpha2)
    a3 = a2 * broken_plaw_norm(lim23, alpha2, alpha3)
    return np.piecewise(Mstar, 
            [
                np.logical_or(Mstar < Ml, Mstar >= Mu),
                np.logical_and(Mstar >= Ml, Mstar < lim01),
                np.logical_and(Mstar >= lim01, Mstar < lim12),
                np.logical_and(Mstar >= lim12, Mstar < lim23),
                np.logical_and(Mstar >= lim23, Mstar < Mu)
            ],
            [
                0., 
                lambda M: a0 * plaw(M, neg_exponent=alpha0),
                lambda M: a1 * plaw(M, neg_exponent=alpha1), 
                lambda M: a2 * plaw(M, neg_exponent=alpha2), 
                lambda M: a3 * plaw(M, neg_exponent=alpha3)
            ])

def integral_plaw(ll, ul, power):
    ''''''
    if ll < ul:
        return int_plaw(ll, ul, power)
    else:
        return 0.
        print('in integral_plaw, ll>=ul')

def normalized(x, func, condition:float=None, *args, **kwargs):
    '''Checks whether or not the mass function has been normalized'''
    if condition:
        if x <= condition:
            return func
        else:
            return 0.
    else:
        return func

def get_norm(normalization, IMF, Mtot:float, min_val:float, max_val:float, *args, **kwargs):
    '''get normalization, where normalization and IMF are both functions '''
    k, M = normalization(IMF, Mtot, min_val, min_val, *args, **kwargs)
    IMF_func = lambda mass: k * IMF(mass, m_max=M, *args, **kwargs)
    IMF_mass_weighted_func = lambda mass: mass_weighted_func(mass, IMF_func,
                                                   *args, **kwargs)
    return k, M, np.vectorize(IMF_func), np.vectorize(IMF_mass_weighted_func)

def normalization_ECMF(ECMF_func, beta, Mtot, lower_lim, upper_lim, Mecl_max_guess=1e1,*args) -> (float, float):
    #print(f'{beta=}, \t{upper_lim=}')
    k_ECMF = lambda x: np.divide(1-beta, np.abs(upper_lim)**(1-beta) - np.abs(x)**(1-beta))
    #def k_ECMF(x):
    #    return np.reciprocal(integral_plaw(x, upper_lim, beta))
    #def k_ECMF(x):
    #    if np.logical_and(x >= lower_lim, x < upper_lim):
    #        print(f'{x=}')
    #        return np.reciprocal(integral_plaw(x, upper_lim, beta))
    #    else:
    #        return 0.
    mass_weighted_ECMF = lambda M_ecl: integral_plaw(lower_lim, upper_lim, beta-1)#mass_weighted_func(M_ecl, ECMF_func)
    func = lambda x: (k_ECMF(x) * mass_weighted_ECMF(x) - Mtot)
    #func = minimize(func=np.multiply(k_ECMF, mass_weighted_ECMF), total=Mtot)
    #print(f'{Mtot=:.2e}')
    #print(f'{lower_lim=:.2e}')
    #print(f'{upper_lim=:.2e}')
    #print(f'{k_ECMF(lower_lim)=:.2e}')
    #print(f'{k_ECMF(lower_lim) * mass_weighted_ECMF(lower_lim)=:.2e}')
    sol = optimize.root_scalar(func, x0=Mecl_max_guess, x1=1e8, rtol=1e-15)
    Mecl_max = sol.root
    #sol = optimize.root(func, Mecl_max_guess, method='hybr')
    #Mecl_max = sol.x[0]
    #print(f'{k_ECMF(Mecl_max)=:.2e}')
    #print(f'{sol.x=}')
    return k_ECMF(Mecl_max), Mecl_max

def normalization_IMF(alpha1, alpha2, alpha3, Mtot, lower_lim, upper_lim
                      ) -> (float, float):
    def k(x):
        if np.logical_and(x >= lower_lim, x < 0.5):
            integral = (2 * integral_plaw(x, 0.5, alpha1) 
                        + integral_plaw(0.5, 1., alpha2) 
                        + integral_plaw(1., upper_lim, alpha3))
            return np.divide(1, integral, where=integral!=0)
        if np.logical_and(x >= 0.5, x < 1.):
            integral = (integral_plaw(x, 1., alpha2) 
                                + integral_plaw(1., upper_lim, alpha3))
            return np.divide(1, integral, where=integral!=0)
        if np.logical_and(x >= 1., x <= upper_lim):
            integral = integral_plaw(x, upper_lim, alpha3)
            return np.divide(1, integral, where=integral!=0)
        else:
            return 0.
    def mass_weighted_IMF(x):
        if np.logical_and(x >= lower_lim, x < 0.5):
            return (2 * integral_plaw(0.08, x, alpha1-1))
        if np.logical_and(x >= 0.5, x < 1.):
            return (2 * integral_plaw(0.08, 0.5, alpha1-1)
                    + integral_plaw(0.5, x, alpha2-1))
        if np.logical_and(x >= 1., x <= upper_lim):
            return (2 * integral_plaw(0.08, 0.5, alpha1-1)
                    + integral_plaw(0.5, 1., alpha2-1)
                    + integral_plaw(1., x, alpha3-1))
        else:
            return 0.
    func = lambda x: (k(x) * mass_weighted_IMF(x) - Mtot)
    try:
        sol = optimize.root_scalar(func, method='bisect', rtol=1e-15,
                                   bracket=(lower_lim, upper_lim))
        m_max = sol.root
        #print(f'{sol.converged=}')
        #sol = optimize.root(func, 10, method='hybr')
        #m_max = sol.x[0]
    except:
        m_max = upper_lim
    #print(f'{Mtot = },\t {m_max = },\t {k(m_max)=}')
    return k(m_max), m_max

def find_closest_prod(number):
    '''
    Auxiliary function for subplots.
    Based on a Vector class resolution (number), adapts the number of subplots
    to fit all of the computed curves. Finds the number of rows and columns 
    (nl, nu) closest to a square that best accomodates the Vector resolution.
    
    number    Integer representing the resolution of the Vector class
    '''
    nl = int(np.floor(np.sqrt(number)))
    if number == np.sign(nl)*np.abs(nl)**2:
        nu = nl
    else:
        nu = nl+1
    while number > nu*nl:
        nu += 1
    return nl, nu


class Downsizing:
    """
    Downsizing relations for ellipticals 
    as it is introduced by Thomas et al. (2005)
    https://ui.adsabs.harvard.edu/abs/2005ApJ...621..673T/abstract
    """
    def __init__(self, M_igal: float) -> None:
        '''
        M_igal             [Msun] 
        downsizing_time    [yr]
        SFR                [Msun/yr]
        '''
        self.M_igal = M_igal
        self.downsizing_time = self.delta_tau(M_igal)
        self.SFR = self.SFR_func(self.M_igal, self.downsizing_time)
        
    def delta_tau(self, M_igal):
        '''
        Returns delta tau in Gyr for the downsizing relation 
        as it is expressed in Recchi+09
        
        M_igal is expressed in Msun and ranges from 1e6 to 1e12
        '''
        return 8.16 * np.e**(-0.556 * np.log10(M_igal) + 3.401) + 0.027       
            
    def SFR_func(self, M_igal, downsizing_time):
        '''SFR [Msun/yr] assuming the downsizing time (Thomas et al., 2005)'''
        return np.divide(M_igal, downsizing_time * 1e9)