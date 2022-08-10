def normalized(x, func, condition=None):
    ''' IMF behavior depending on whether or not it has been normalized '''
    if condition:
        if x <= condition:
            return func
        else:
            return 0.
    else:
        return func
    
def normalization(IMF, M, lower_lim, upper_lim, **kwargs) -> (float, float):
    r'''
    Function that extracts k and m_max (blue boxes in the notes)
    IMF:    mass distribution function, i.e. either Eq. (1) or (8)
    M:      value of the integrals in Eq. (2) and (9)
    k:      normalization of the IMF function, i.e. Eq. (3) and (10)
    upper_lim and lower_lim:    
            global upper and lower limits on the integrals
    guess:  guess on the local upper (lower) limit on the integrals 
            of Eq.2 and Eq.9 (of Eq.3 and Eq.10)
    x:      evaluated local upper (lower) limit on the integrals
            of Eq.2 and Eq.9 (of Eq.3 and Eq.10)
    **kwargs    optional keyword arguments
    
    -----
    
    .. math::
    `M = \int_{\mathrm{lower_lim}}^{\mathrm{m_max}}
    m \, \mathrm{IMF}(m,...)\,\mathrm{d}m`
    
    .. math::
    `1 = \int_{\mathrm{m_max}}^{{\rm upper_lim}}{\mathrm{IMF}(m,...)} \,\mathrm{d}m`
    '''
    k = lambda x: np.reciprocal(integr.quad(IMF, x, upper_lim, 
                                    epsrel=1e-8, limit=int(1e3), maxp1=int(1e3), limlst=int(1e3))[0])
    def weighted_IMF(m, x):
        return m * IMF(m) * k(x)
    #weighted_IMF = lambda m: m * IMF(m)
    func = lambda x: (integr.quad(weighted_IMF, lower_lim, x, args=(x,), 
                                    epsrel=1e-8, limit=int(1e3), maxp1=int(1e3), limlst=int(1e3))[0] - M)
    sol = optimize.root_scalar(func, bracket=[lower_lim, upper_lim], rtol=1e-8)
    m_max = sol.root
    return k(m_max), m_max

def normalization_IMF(IMF, M, lower_lim, upper_lim, alpha_3):
    '''duplicate of normalization !!!!!!! '''
    k = lambda x: np.reciprocal(integr.quad(IMF, x, upper_lim, args=(alpha_3,))[0])
    def weighted_IMF(m, x, alpha_3):
        return m * IMF(m, alpha_3=alpha_3) * k(x)
    func = lambda x: integr.quad(weighted_IMF, lower_lim, x, args=(x,alpha_3))[0] - M
    sol = optimize.root_scalar(func, bracket=[lower_lim, upper_lim])
    m_max = sol.root
    return k(m_max), m_max