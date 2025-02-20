import numpy as np
import scipy.integrate as integr
from scipy import optimize
from typing import Optional
from typing import Callable
from typing import Any

'''
This module contains utility functions used by the pyIGIMF package
'''

def mass_weighted_func(M, func, *args, **kwargs):
    """
    Apply a given function to mass values and return the mass-weighted result.

    Parameters
    ----------
    M : array-like
        The mass values to be processed.
    func : callable
        The function to apply to the mass values. It should accept `M` as its first parameter.
    *args : tuple
        Additional positional arguments to pass to `func`.
    **kwargs : dict
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    array-like
        The result of multiplying the mass values by the result of `func`.
    """
    return np.multiply(M, func(M, *args, **kwargs))

def powerlaw(M, alpha=2.3):
    """
    A simple power law function.

    Parameters
    ----------
    M : float
        The mass of the object belonging to a power law distribution.
    alpha : float, optional
        The exponent of the power law. Defaults to 2.3.

    Returns
    -------
    M**(-alpha) : float
        The result of the power law.
    """
    return M**(-alpha)

def broken_powerlaw(Mstar, a1, a2, a3, sIMF_params = {
                            'alpha1': 1.3,
                            'alpha2': 2.3,
                            'alpha3': 2.3,
                            'Ml': 0.08,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': 150
}):
    """
    A broken powerlaw function, used to describe the IMF. 
    This function is not normalized (k=1)
    
    The piecewise function is constructed with the following parts:
       1. The first part is zero, for masses less than Ml or greater than Mu.
       2. The second part is a power law with exponent alpha1, for masses between Ml and lim12.
       3. The third part is a power law with exponent alpha2, for masses between lim12 and lim23.
       4. The fourth part is a power law with exponent alpha3, for masses between lim23 and Mu.

    Parameters
    ----------
    Mstar : float
        The mass of the star.
    a1 : float
        The normalisation constant of the first power law.
    a2 : float
        The normalisation constant of the second power law.
    a3 : float
        The normalisation constant of the third power law.
    sIMF_params : dict, optional
        The parameters of the IMF. Defaults to values from Kroupa (2001).
    
    sIMF_params['Mu'] is augmented by 1e-13 to ensure the IMF is defined on the last item too

    Returns
    -------
    float
        The result of the broken power law.
    """
    return np.piecewise(np.float64(Mstar), 
            [
                np.logical_or(Mstar < sIMF_params['Ml'], Mstar >= sIMF_params['Mu']+1e-13),
                np.logical_and(Mstar >= sIMF_params['Ml'], Mstar < sIMF_params['Mlim12']),
                np.logical_and(Mstar >= sIMF_params['Mlim12'], Mstar < sIMF_params['Mlim23']),
                np.logical_and(Mstar >= sIMF_params['Mlim23'], Mstar < sIMF_params['Mu']+1e-13)
            ],
            [
                0., 
                lambda M: a1 * powerlaw(M, alpha=sIMF_params['alpha1']), 
                lambda M: a2 * powerlaw(M, alpha=sIMF_params['alpha2']), 
                lambda M: a3 * powerlaw(M, alpha=sIMF_params['alpha3'])
            ])

def powerlaw_continuity(limit, lower_exp, upper_exp):
    """
    Ensure continuity in the broken power law.

    Parameters
    ----------
    limit : float
        The mass limit between the two power laws.
    lower_exp : float
        The exponent of the lower mass power law.
    upper_exp : float
        The exponent of the upper mass power law.

    Returns
    -------
    norm : float
        The normalisation factor to ensure continuity.
    """
    return limit**(upper_exp - lower_exp)

def IMF_normalization_constants(os_norm=1, norm_wrt=150, sIMF_params={
                            'alpha1': 1.3,
                            'alpha2': 2.3,
                            'alpha3': 2.3,
                            'Ml': 0.08,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': 150
                        }):
    # The normalisation constants are calculated such that the function is continuous.
    a1 = os_norm 
    a2 = a1 * powerlaw_continuity(sIMF_params['Mlim12'], sIMF_params['alpha1'], sIMF_params['alpha2'])
    a3 = a2 * powerlaw_continuity(sIMF_params['Mlim23'], sIMF_params['alpha2'], sIMF_params['alpha3'])
    
    # The IMF function is calculated using the broken power law.
    IMF_func = lambda Mstar: broken_powerlaw(Mstar, a1, a2, a3, sIMF_params=sIMF_params)
    
    # The normalisation constant is calculated by integrating the IMF function from 0.08 to 150 Msun.
    N_norm = a3 * IMF_func(norm_wrt)
    
    # The normalisation constants are adjusted such that the integral of the IMF function is 1.
    a1 /= N_norm
    a2 /= N_norm
    a3 /= N_norm 
    
    return a1, a2, a3

def Kroupa01(norm_wrt = 150., os_norm=1., sIMF_params = {
                            'alpha1': 1.3,
                            'alpha2': 2.3,
                            'alpha3': 2.3,
                            'Ml': 0.08,
                            'Mlim12': 0.5,
                            'Mlim23': 1.,
                            'Mu': 150
                        },
            ):
    '''
    IMF as a broken power law, from Kroupa (2001).
    https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K/abstract 

    The parameters are adjustable inside of the sIMF_params dictionary.
    
    norm_wrt is the mass at which the normalisation is 1. Defaults to 150 Msun.
    norm_wrt helps when using this function manually. Valid values are:
        Mlim23 < norm_wrt < Mu
    os_norm is the normalisation of the optimal IMF, used in the StellarIMF class.
    '''
    a1, a2, a3 = IMF_normalization_constants(os_norm=os_norm, norm_wrt=norm_wrt)
    
    # The final IMF function is returned.
    return lambda Mstar: broken_powerlaw(Mstar, a1, a2, a3, sIMF_params=sIMF_params)

def integrate_powerlaw(lower_limit, upper_limit, exponent, weighted=False):
    """
    Integrate a power law function from a lower limit to an upper limit.

    .. math::
        I = \frac{upper_limit^{1-\text{exponent}} - lower_limit^{1-\text{exponent}}}{1-\text{exponent}}

    Parameters
    ----------
    lower_limit : float
        The lower limit of the integral.
    upper_limit : float
        The upper limit of the integral.
    exponent : float
        The exponent of the power law.

    Returns
    -------
    float
        The result of the integral.
    """
    if weighted == False:
        power = 1
    else:
        power = 2
    if np.isclose(power-exponent, 0):
        return np.log(upper_limit) - np.log(lower_limit)
    else:
        return (upper_limit**(power-exponent) - lower_limit**(power-exponent)) / (power-exponent)

def integral_powerlaw(ll, ul, power):
    """
    Return the integral of a simple power law x**(-power) from ll to ul.
    
    Parameters
    ----------
    ll : float
        The lower limit of the integral.
    ul : float
        The upper limit of the integral.
    power : float
        The exponent of the power law.
    
    Returns
    -------
    float
        The result of the integral.
    """
    if ll < ul:
        return integrate_powerlaw(ll, ul, power)
    else:
        return 0.
    
def normalization_check(x: float, func: float #Callable[..., Any]
                        , condition: Optional[float]=None):
    """
    If condition is None, return the function.
    Otherwise, impose the upper limit ` on the function.
    Checks whether or not the mass function has been normalized.
    If not, returns the function. 
    If yes, return the function up until the upper limit, otherwise return 0.
    
    Parameters
    ----------
    x : float
        The mass value to be processed.
    func : float
        The float resulting from the application of the function to x.
    condition : float, optional
        The mass limit above which the mass function should return 0. Default is None.
    *args : tuple
        Additional positional arguments to pass to `func`.
    **kwargs : dict
        Additional keyword arguments to pass to `func`.
    
    Returns
    -------
    float
        The normalized mass function.
    """
    if condition:
        if x < condition or np.isclose(x, condition, atol=1e-8):
            return func
        else:
            return 0.
    else:
        return func
    
def optimal_sampling_ECMF(beta, Mtot, lower_lim, upper_lim):
    """
    Calculate the normalization constant of the Embedded Cluster Mass Function (ECMF).

    This function determines the normalization constant and the effective upper limit 
    of the ECMF using numerical methods. It is based on a power-law distribution, 
    with special cases handled for specific values of the power-law exponent.

    Parameters
    ----------
    beta : float
        The exponent of the power law for the ECMF.
    Mtot : float
        The total mass of the system.
    lower_lim : float
        The lower limit of the mass range.
    upper_lim : float
        The theoretical upper limit of the mass range.

    Returns
    -------
    tuple
        A tuple containing the normalization constant and the effective upper limit
        of the ECMF.

    Notes
    -----
    The function includes special handling for when the exponent `beta` equals 1 or 2, 
    optimizing the calculations for these cases by using logarithmic integrals.
    """

    def k_ECMF(x):
        k_ECMF = integrate_powerlaw(x, upper_lim, beta, weighted=False)
        return np.reciprocal(k_ECMF)
    
    def minimization_func(x, lower_lim, beta, Mtot, upper_lim):
        """
        The function to be minimized to find the normalization constant of the ECMF.
        
        Parameters
        ----------
        x : float
            The current guess for the upper limit.
        lower_lim : float
            The lower limit of the ECMF.
        beta : float
            The exponent of the power law.
        Mtot : float
            The total mass of the system.
        upper_lim : float
            The theoretical upper limit of the ECMF.
        
        Returns
        -------
        float
            The difference between the total mass and the integral of the ECMF.
        """
        I1 = integrate_powerlaw(lower_lim, x, beta, weighted=True)
        I2 = integrate_powerlaw(x, upper_lim, beta, weighted=False)
        return Mtot * I2 - I1
    
    def solve_x(lower_lim, beta, Mtot, upper_lim, x_guess=None):

        # Solve numerically (bisection)
        x_solution = optimize.bisect(minimization_func, lower_lim, upper_lim, args=(lower_lim, beta, Mtot, upper_lim), xtol=1e-20)
        
        # Ensure x is within valid range (a < x < U)
        if lower_lim <= x_solution <= upper_lim:
            return x_solution
        else:
            raise ValueError(f"No valid solution found within the range ({lower_lim}, {upper_lim}) during the ECMF optimal sampling.")
    
    def execute(lower_lim, beta, Mtot, upper_lim):
        try:
            real_upper_limit = solve_x(lower_lim, beta, Mtot, upper_lim)
            k = k_ECMF(real_upper_limit)
            #print(f'{Mtot = }, {real_upper_limit/upper_lim = }, {k = }')
            return k, real_upper_limit
        except ValueError as e:
            print(e)

    return execute(lower_lim, beta, Mtot, upper_lim)

def optimal_sampling_IMF(M_ecl, IGIMF_params):
    
    def k_IMF(x):
        if np.logical_and(x >= IGIMF_params['Ml'], x <= IGIMF_params['Mu']):
            k_IMF = IMF_integrals(x)
            return np.reciprocal(k_IMF)
        else:
            return 0.
    
    def IMF_integrals(x, weighted=False):
        # if np.logical_and(x >= IGIMF_params['Ml'], x < IGIMF_params['Mlim12']):
        #     I = (IGIMF_params['a1'] * integrate_powerlaw(x, IGIMF_params['Mlim12'], IGIMF_params['alpha1'], weighted=weighted) 
        #         +IGIMF_params['a2'] * integrate_powerlaw(IGIMF_params['Mlim12'], IGIMF_params['Mlim23'], IGIMF_params['alpha2'], weighted=weighted) 
        #         +IGIMF_params['a3'] * integrate_powerlaw(IGIMF_params['Mlim23'], IGIMF_params['Mu'], IGIMF_params['alpha3'], weighted=weighted))
        # elif np.logical_and(x >= IGIMF_params['Mlim12'], x < IGIMF_params['Mlim23']):
        #     I = (IGIMF_params['a2'] * integrate_powerlaw(x, IGIMF_params['Mlim23'], IGIMF_params['alpha2'], weighted=weighted) 
        #         +IGIMF_params['a3'] * integrate_powerlaw(IGIMF_params['Mlim23'], IGIMF_params['Mu'], IGIMF_params['alpha3'], weighted=weighted))
        # elif np.logical_and(x >= IGIMF_params['Mlim23'], x <= IGIMF_params['Mu']):
        #     I = IGIMF_params['a3'] * integrate_powerlaw(x, IGIMF_params['Mu'], IGIMF_params['alpha3'], weighted=weighted)
        # else:
        #     I = 0.
        if np.logical_and(x >= IGIMF_params['Ml'], x < IGIMF_params['Mlim12']):
            I = (2 * integrate_powerlaw(x, IGIMF_params['Mlim12'], IGIMF_params['alpha1'], weighted=weighted) 
                +1 * integrate_powerlaw(IGIMF_params['Mlim12'], IGIMF_params['Mlim23'], IGIMF_params['alpha2'], weighted=weighted) 
                +1 * integrate_powerlaw(IGIMF_params['Mlim23'], IGIMF_params['Mu'], IGIMF_params['alpha3'], weighted=weighted))
        elif np.logical_and(x >= IGIMF_params['Mlim12'], x < IGIMF_params['Mlim23']):
            I = (1 * integrate_powerlaw(x, IGIMF_params['Mlim23'], IGIMF_params['alpha2'], weighted=weighted) 
                +1 * integrate_powerlaw(IGIMF_params['Mlim23'], IGIMF_params['Mu'], IGIMF_params['alpha3'], weighted=weighted))
        elif np.logical_and(x >= IGIMF_params['Mlim23'], x <= IGIMF_params['Mu']):
            I = 1 * integrate_powerlaw(x, IGIMF_params['Mu'], IGIMF_params['alpha3'], weighted=weighted)
        else:
            I = 0.
        return I
        
    def weighted_IMF_integrals(x, weighted=True):
        # if np.logical_and(x >= IGIMF_params['Mlim23'], x <= IGIMF_params['Mu']):
        #     I = (IGIMF_params['a1'] * integrate_powerlaw(IGIMF_params['Ml'], IGIMF_params['Mlim12'], IGIMF_params['alpha1'], weighted=weighted) 
        #         +IGIMF_params['a2'] * integrate_powerlaw(IGIMF_params['Mlim12'], IGIMF_params['Mlim23'], IGIMF_params['alpha2'], weighted=weighted) 
        #         +IGIMF_params['a3'] * integrate_powerlaw(IGIMF_params['Mlim23'], x, IGIMF_params['alpha3'], weighted=weighted))
        # elif np.logical_and(x >= IGIMF_params['Mlim12'], x < IGIMF_params['Mlim23']):
        #     I = (IGIMF_params['a1'] * integrate_powerlaw(IGIMF_params['Ml'], IGIMF_params['Mlim12'], IGIMF_params['alpha1'], weighted=weighted) 
        #         +IGIMF_params['a2'] * integrate_powerlaw(IGIMF_params['Mlim12'], x, IGIMF_params['alpha2'], weighted=weighted))
        # elif np.logical_and(x >= IGIMF_params['Ml'], x < IGIMF_params['Mlim12']):
        #     I = IGIMF_params['a1'] * integrate_powerlaw(IGIMF_params['Ml'], x, IGIMF_params['alpha1'], weighted=weighted)
        # else:
        #     I = 0.
        if np.logical_and(x >= IGIMF_params['Mlim23'], x <= IGIMF_params['Mu']):
            I = (2 * integrate_powerlaw(IGIMF_params['Ml'], IGIMF_params['Mlim12'], IGIMF_params['alpha1'], weighted=weighted) 
                +1 * integrate_powerlaw(IGIMF_params['Mlim12'], IGIMF_params['Mlim23'], IGIMF_params['alpha2'], weighted=weighted) 
                +1 * integrate_powerlaw(IGIMF_params['Mlim23'], x, IGIMF_params['alpha3'], weighted=weighted))
        elif np.logical_and(x >= IGIMF_params['Mlim12'], x < IGIMF_params['Mlim23']):
            I = (2 * integrate_powerlaw(IGIMF_params['Ml'], IGIMF_params['Mlim12'], IGIMF_params['alpha1'], weighted=weighted) 
                +1 * integrate_powerlaw(IGIMF_params['Mlim12'], x, IGIMF_params['alpha2'], weighted=weighted))
        elif np.logical_and(x >= IGIMF_params['Ml'], x < IGIMF_params['Mlim12']):
            I = 2 * integrate_powerlaw(IGIMF_params['Ml'], x, IGIMF_params['alpha1'], weighted=weighted)
        else:
            I = 0.
        return I
    
    def minimization_func(x, M_ecl):
        """
        The function to be minimized to find the normalization constant of the ECMF.
        
        Parameters
        ----------
        x : float
            The current guess for the upper limit.
        lower_lim : float
            The lower limit of the ECMF.
        beta : float
            The exponent of the power law.
        Mtot : float
            The total mass of the system.
        upper_lim : float
            The theoretical upper limit of the ECMF.
        
        Returns
        -------
        float
            The difference between the total mass and the integral of the ECMF.
        """
        I1 = weighted_IMF_integrals(x, weighted=True)
        I2 = IMF_integrals(x, weighted=False)
    
        return M_ecl * I2 - I1
    
    def solve_x_snake(Mtot, lower_lim, upper_lim):

        # Solve numerically (bisection)
        x_solution = optimize.bisect(minimization_func, lower_lim, upper_lim, args=(Mtot), xtol=1e-20)
        
        # Ensure x is within valid range (a < x < U)
        if lower_lim <= x_solution <= upper_lim:
            return x_solution
        else:
            raise ValueError(f"No valid solution found within the range ({lower_lim}, {upper_lim}).")
    
    def solve_x(Mtot, lower_lim, upper_lim):
        try:
            sol = optimize.root_scalar(minimization_func, method='bisect', rtol=1e-20, args=(Mtot), bracket=(lower_lim, upper_lim))
            m_max = sol.root
        except:
            m_max = upper_lim
        return m_max
    
    def execute(Mtot, lower_lim, upper_lim):
        try:
            real_upper_limit = solve_x(Mtot, lower_lim, upper_lim)
            k = k_IMF(real_upper_limit)
            print(f'{Mtot = }, {real_upper_limit/upper_lim = }, {k = }')
            return k, real_upper_limit
        except ValueError as e:
            print(e)

    return execute(M_ecl, IGIMF_params['Ml'], IGIMF_params['Mu'])
    





def find_closest_factors(number):
    """
    Find the pair of factors closest to a given number.
    Function used in plots.py to rescale the number of subplots

    Parameters
    ----------
    number : int
        The number to find the closest factors of.

    Returns
    -------
    lower_factor : int
        The lower of the two factors.
    upper_factor : int
        The upper of the two factors.
    """

    lower_factor = int(np.floor(np.sqrt(number)))
    if number == lower_factor ** 2:
        upper_factor = lower_factor
    else:
        upper_factor = lower_factor + 1

    while number > upper_factor * lower_factor:
        upper_factor += 1

    return lower_factor, upper_factor
    


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


def normalized(x, func, condition=None, *args, **kwargs):
    ''' IMF behavior depending on whether or not it has been normalized '''
    if condition:
        if x <= condition:
            return func
        else:
            return 0.
    else:
        return func