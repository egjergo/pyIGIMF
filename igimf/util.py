import numpy as np
from scipy import optimize
import scipy.integrate as integr

def find_closest_prod(number):
    nl = int(np.floor(np.sqrt(number)))
    if number == nl**2:
        nu = nl
    else:
        nu = nl+1
    while number > nu*nl:
        nu += 1
    return nl, nu
       
def weighted_func(M, func, *args, **kwargs):
    return np.multiply(M, func(M, *args, **kwargs))

def int_plaw(ll, ul, power):
    return np.divide(ul**(1-power) - ll**(1-power), 1-power)

def integral_powerlaw(ll, ul, power):
    if ll < ul:
        return int_plaw(ll, ul, power)
    else:
        return 0

def normalized(x, func, condition=None, *args, **kwargs):
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
    IMF_weighted_func = lambda mass: weighted_func(mass, IMF_func,
                                                   *args, **kwargs)
    return k, M, np.vectorize(IMF_func), np.vectorize(IMF_weighted_func)

def normalization_ECMF(ECMF, beta, Mtot, lower_lim, upper_lim, *args) -> (float, float):
    k_ECMF = lambda x: np.divide(1-beta, upper_lim**(1-beta) - x**(1-beta))
    weighted_ECMF = lambda x: integral_powerlaw(lower_lim, upper_lim, beta-1)
    func = lambda x: (k_ECMF(x) * weighted_ECMF(x) - Mtot)
    sol = optimize.root_scalar(func, x0=1e0, x1=1e5, rtol=1e-8)
    Mecl_max = sol.root
    return k_ECMF(Mecl_max), Mecl_max
        
def normalization_IMF(alpha1, alpha2, alpha3, Mtot, lower_lim, upper_lim) -> (float, float):
    def k(x):
        if np.logical_and(x >= lower_lim, x < 0.5):
            return (np.reciprocal(2 * integral_powerlaw(x, 0.5, alpha1) 
                                + integral_powerlaw(0.5, 1., alpha2) 
                                + integral_powerlaw(1., upper_lim, alpha3)))
        if np.logical_and(x >= 0.5, x < 1.):
            return (np.reciprocal(integral_powerlaw(x, 1., alpha2) 
                                + integral_powerlaw(1., upper_lim, alpha3)))
        if np.logical_and(x >= 1., x <= upper_lim):
            return (np.reciprocal(integral_powerlaw(x, upper_lim, alpha3)))
        else:
            return 0.
    def weighted_IMF(x):
        if np.logical_and(x >= lower_lim, x < 0.5):
            return (2 * integral_powerlaw(0.08, x, alpha1-1))
        if np.logical_and(x >= 0.5, x < 1.):
            return (2 * integral_powerlaw(0.08, 0.5, alpha1-1)
                    + integral_powerlaw(0.5, x, alpha2-1))
        if np.logical_and(x >= 1., x <= upper_lim):
            return (2 * integral_powerlaw(0.08, 0.5, alpha1-1)
                    + integral_powerlaw(0.5, 1., alpha2-1)
                    + integral_powerlaw(1., x, alpha3-1))
        else:
            return 0.
    func = lambda x: (k(x) * weighted_IMF(x) - Mtot)
    try:
        sol = optimize.root_scalar(func, method='bisect', rtol=1e-15,
                                   bracket=(lower_lim+.01, upper_lim-.01))
        m_max = sol.root
        if m_max > upper_lim:
            m_max = upper_lim
    except:
        m_max = upper_lim
    #print(f'{Mtot = },\t {m_max = },\t {k(m_max)=}')
    return k(m_max), m_max


    def get_lists(self):
        IMF_v_Z_list = []
        alpha1_Z_list = []
        alpha1_Z_list = []
        m_max_Z_list = []
        k_IMF_Z_list = []
        for Z in Z_massfrac_v:
            IMF_v_list = []
            alpha1_list = []
            alpha3_list = []
            m_max_list = []
            k_IMF_list = []
            for M in Mecl_v:
                igimf4 = IGIMF4.IGIMF(Z, downsizing_obj.SFR)
                sIMF = igimf4.stellar_IMF(M)
                #print (f"M=%.2e,\t alpha1=%.2f,\t alpha2=%.2f,\t alpha3=%.2f,\t m_max = %.2e,\t [Z] = %.2f"%(M, sIMF[4], sIMF[5], sIMF[6], sIMF[1], igimf4.metallicity))
                #IMF_v = sIMF[2](mstar_v)
                alpha1_list.append(sIMF[4])
                alpha3_list.append(sIMF[6])
                m_max_list.append(sIMF[1])
                k_IMF_list.append(sIMF[0])
                IMF_v_list.append(IMF_v)
                #igimf4.ECMF_plot(Mecl_v, ECMF_v)
            IMF_v_Z_list.append(IMF_v_list)
            alpha1_Z_list.append(alpha1_list)
            alpha3_Z_list.append(alpha3_list)
            m_max_Z_list.append(m_max_list)
            k_IMF_Z_list.append(k_IMF_list)
        return k_IMF_Z_list, m_max_Z_list, IMF_v_Z_list, alpha1_Z_list, alpha1_Z_list
        
        
class Downsizing:
    """Downsizing relations as introduced by Thomas et al. (2005)"""
    
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