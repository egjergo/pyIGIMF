import numpy as np
import astropy.units as u


class Downsizing:
    """
    Downsizing relations as introduced by Thomas et al. (2005)
    
    .. code-block:: python
        # Create a new instance of the Downsizing class with M_igal = 1e10 Msun
        downsizing_example = Downsizing(M_igal=1e10)

        # Access the variables
        downsizing_time_example = downsizing_example.SFR
        SFR_example = downsizing_example.SFR
        M_igal_example = downsizing_example.M_igal
        
        # Access their values (for computations)
        print(downsizing_time_example.value)
        print(SFR_example.value)
        print(M_igal_example.value) 
        
        # Access their units
        print(downsizing_time_example.unit)
        print(SFR_example.unit)
        print(M_igal_example.unit)
    
    """
    
    def __init__(self, M_igal: float) -> None:
        '''
        M_igal             [Msun] 
        downsizing_time    [Gyr]
        SFR                [Msun/yr]
        '''
        self._M_igal = M_igal * u.Msun
        self._downsizing_time = self.delta_tau(M_igal) * u.Gyr
        self._SFR = self.SFR_func(self._M_igal, self._downsizing_time) / u.yr
        self.M_igal = self.repr('M_igal')
        self.downsizing_time = self.repr('downsizing_time')
        self.SFR = self.repr('SFR')

    #@property
    def repr(self, var: str):
        variable = getattr(self,'_'+var)
        print(f"{variable.value:.2e} {variable.unit}")
        return variable


    def __repr__(self):
        """Unambiguous representation (used for debugging)."""
        attributes = [
            f"{key}={value.value:.2e} {value.unit}" 
            for key, value in vars(self).items() 
            if isinstance(value, u.Quantity)
        ]
        return f"Downsizing({', '.join(attributes)})"

    def __str__(self):
        """User-friendly representation (used for printing)."""
        lines = [
            f"{key.replace('_', ' ').capitalize()}: {value.value:.2e} {value.unit}" 
            for key, value in vars(self).items() 
            if isinstance(value, u.Quantity)
        ]
        return "\n".join(lines)

    def delta_tau(self, M_igal):
        '''
        Returns delta tau in Gyr for the downsizing relation 
        as it is expressed in Recchi+09
        
        M_igal is expressed in Msun and ranges from 1e6 to 1e12
        '''
        return 8.16 * np.e**(-0.556 * np.log10(M_igal) + 3.401) + 0.027       
            
    def SFR_func(self, M_igal, downsizing_time):
        '''SFR [Msun/yr] assuming the downsizing time (Thomas et al., 2005)'''
        return np.divide(M_igal, downsizing_time.value * 1e9)