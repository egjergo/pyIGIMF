import numpy as np
import pandas as pd
from igimf import instance as inst
from igimf import util as utl
from igimf import plots as plts

class Vectors:
    '''
    Set up the vectors in order to compute the IGIMF instances
    across a range of (t, SFR(t), Z(t))
    '''
    def __init__(self, resolution=30):
        
        # Setup and parameters
        self.plots = plts.Plots()
        self.resolution = resolution
        par = inst.Parameters(0.01, 1) # dummy values to extract param. below:
        self.solar_metallicity = par.solar_metallicity
        self.m_star_max = par.m_star_max
        self.m_star_min = par.m_star_min
        
        # Vectors
        self.M_igal_v = self.logspace_v_func(resolution, minlog=6, maxlog=11)
        self.Mecl_v = self.logspace_v_func(resolution, minlog=np.log10(5), maxlog=10)
        self.Mecl_v = np.array([5.00000000e+00, 1.e+03, 1.e+06, 1.00000000e+10])
        self.Z_massfrac_v = self.logspace_v_func(resolution, minlog=-7, maxlog=1)#, maxlog=.5) 
        self.Z_massfrac_v *= self.solar_metallicity # to make subplots labels consistent
        self.mstar_v = np.logspace(np.log10(self.m_star_min),
                                   np.log10(self.m_star_max-0.1), num=100)
        self.SFR_v = self.logspace_v_func(resolution, minlog=-5.5, maxlog=4)
        self.metallicity_v = np.log10(self.Z_massfrac_v/self.solar_metallicity)
        #self.metallicity_v = np.array([-4, -1., 0., .5])
        self.Z_massfrac_v = np.power(10., self.metallicity_v) * self.solar_metallicity
        
    def logspace_v_func(self, res, minlog=-1, maxlog=1):
        return np.logspace(minlog, maxlog, num=res)
        
        
class SingleECMF(Vectors):
    """
    A single instance of ECMF.
    
    Requires either
    M_igal (to extract the SFR through Downsizing)
    or SFR directly.
    """
    def __init__(self, M_igal=None, SFR=None, metal_mass_fraction=None, downsizing_bool=False):
        super().__init__()
        if downsizing_bool == True:
            if M_igal is not None and SFR is not None:
                raise ValueError("Define only one between M_igal and SFR")
            elif M_igal:
                self.M_igal = M_igal
                self.downsizing_obj = utl.Downsizing(M_igal)
                self.SFR = self.downsizing_obj.SFR
            elif SFR:
                self.SFR = SFR
        else:
            self.SFR = SFR
        self.metal_mass_fraction = metal_mass_fraction
        self.o_ECMF = inst.ECMF(SFR=self.SFR, metal_mass_fraction=self.metal_mass_fraction)
        self.__dict__.update(self.o_ECMF.__dict__)
        return None

    def ECMF_plot(self):
        ECMF_v = self.ECMF_func(self.Mecl_v)
        self.plots.ECMF_plot(self.Mecl_v, ECMF_v, self.SFR)

class ECMFbySFR(SingleECMF):
    '''
    Compute various ECMFs along various SFR(t) 
    (e.g., use SFR_v from the Vectors superclass)
    '''
    def __init__(self, M_igal=None, SFR=None, metal_mass_fraction=None):
        if M_igal is None and SFR is None:
            M_igal = 1e10 #M_igal = 1e10, a dummy
        super().__init__(M_igal=M_igal, SFR=SFR, metal_mass_fraction=metal_mass_fraction) 
        returned_lists = self.return_lists()
        self.beta_ECMF_list = returned_lists[0]
        self.MeclMax_list = returned_lists[1]
        self.ECMF_v_list = returned_lists[2]
        return None
    
    def return_lists(self):
        beta_ECMF_list, MeclMax_list, ECMF_v_list = [], [], []
        for S in self.SFR_v:
            o_ECMF = SingleECMF(SFR=S, metal_mass_fraction=self.metal_mass_fraction)
            beta_ECMF_list.append(o_ECMF.beta_ECMF)
            MeclMax_list.append(o_ECMF.M_max)
            ECMF_v = o_ECMF.ECMF_func(self.Mecl_v)
            ECMF_v_list.append(ECMF_v)
        return beta_ECMF_list, MeclMax_list, ECMF_v_list
    
    def all_plots(self):
        self.beta_ECMF_bySFR_plot()
        self.MeclMax_bySFR_plot()
        self.Mecl_power_beta_plot()
        self.ECMF_plots()
    
    def beta_ECMF_bySFR_plot(self):
        self.plots.beta_ECMF_bySFR_plot(self.SFR_v, self.beta_ECMF_list)
    
    def MeclMax_bySFR_plot(self):
        self.plots.MeclMax_bySFR_plot(self.SFR_v, self.MeclMax_list)
    
    def Mecl_power_beta_plot(self):
        self.plots.Mecl_power_beta_plot(self.Mecl_v, self.beta_ECMF_list)
    
    def ECMF_plots(self):
        self.plots.ECMF_plots(self.Mecl_v, self.ECMF_v_list, self.SFR_v)
   
           
class SingleStellarIMF(Vectors):
    def __init__(self, M_ecl, metal_mass_fraction, SFR):
        super().__init__()
        self.M_ecl = M_ecl
        self.metal_mass_fraction = metal_mass_fraction
        self.SFR = SFR
        self.o_IMF = inst.StellarIMF(M_ecl, metal_mass_fraction, SFR)
        self.__dict__.update(self.o_IMF.__dict__)
        self.IMF_v = self.IMF_func(self.mstar_v)
        self.IMF_mass_weighted_v = self.IMF_mass_weighted_func(self.mstar_v)
        return None
    
    def IMF_plot(self):
        self.plots.IMF_plot(self.mstar_v, self.IMF_v, self.M_ecl, 
                            self.metallicity, self.SFR)
 

class SNCC_vs_lowmass:
    def __init__(self, sIMF_instance: SingleStellarIMF):
        for item in list(sIMF_instance.__dict__):
            self.__dict__[item] = sIMF_instance.__dict__[item] 
 
   
class StellarIMFbyMtot(SingleStellarIMF):
    def __init__(self, metal_mass_fraction, SFR, M_ecl=1e5):
        super().__init__(M_ecl, metal_mass_fraction, SFR)
        self.return_lists()
        self.k_idx = np.where(np.array(self.k_IMF_list)<=1e10)[0]
    
    def return_lists(self):
        k_IMF_list, m_max_list, IMF_v_list = [], [], []
        for M in self.Mecl_v:
            imf = SingleStellarIMF(M, self.metal_mass_fraction, self.SFR)
            k_IMF_list.append(imf.k_star)
            m_max_list.append(imf.m_max)
            IMF_v_list.append(imf.IMF_v)
        self.k_IMF_list = k_IMF_list
        self.m_max_list = m_max_list
        self.IMF_v_list = IMF_v_list
    
    def IMF_plots(self):
        return self.plots.IMF_plots(self.mstar_v, self.IMF_v_list,
                        self.Mecl_v, self.k_idx, self.metal_mass_fraction)

class StellarmmaxMecl(SingleStellarIMF):
    def __init__(self, metal_mass_fraction, SFR, M_ecl=1e5):
        super().__init__(M_ecl, metal_mass_fraction, SFR)
        self.k_IMF_Z_list, self.m_max_Z_list, self.IMF_v_Z_list = self.return_lists()
    
    def return_lists(self):
        IMF_v_Z_list = []
        alpha1_Z_list = []
        alpha3_Z_list = []
        m_max_Z_list = []
        k_IMF_Z_list = []
        for Z in self.Z_massfrac_v:
            IMF_v_list = []
            alpha1_list = []
            alpha3_list = []
            m_max_list = []
            k_IMF_list = []
            for M in self.Mecl_v:
                imf = SingleStellarIMF(M, Z, self.SFR)
                k_IMF_list.append(imf.k_star)
                m_max_list.append(imf.m_max)
                IMF_v_list.append(imf.IMF_v)
                #print (f"M=%.2e,\t alpha1=%.2f,\t alpha2=%.2f,\t alpha3=%.2f,\t m_max = %.2e,\t [Z] = %.2f"%(M, sIMF[4], sIMF[5], sIMF[6], sIMF[1], igimf4.metallicity))
                #IMF_v = sIMF[2](mstar_v)
                #alpha1_list.append(imf.alpha1)
                #alpha3_list.append(imf.alpha3)
                #igimf4.ECMF_plot(Mecl_v, ECMF_v)
            IMF_v_Z_list.append(IMF_v_list)
            #alpha1_Z_list.append(alpha1_list)
            #alpha3_Z_list.append(alpha3_list)
            m_max_Z_list.append(m_max_list)
            k_IMF_Z_list.append(k_IMF_list)
        return k_IMF_Z_list, m_max_Z_list, IMF_v_Z_list#, alpha1_Z_list, alpha3_Z_list
 
    def k_Z_plot(self):
        return self.plots.k_Z_plot(self.metallicity_v, self.k_IMF_Z_list, 
                                   self.m_max_Z_list, self.Mecl_v)        

class StellarIMFbyZbyMecl(SingleStellarIMF):
    def __init__(self, SFR, M_ecl=1e5, metal_mass_fraction=0.1*0.0134):
        super().__init__(M_ecl, metal_mass_fraction, SFR)
        self.IMF_Z_v_list, self.mw_IMF_Z_v_list = self.return_list()
    
    def return_list(self):
        IMF_Z_v_list = []
        mw_IMF_Z_v_list = []
        for M in self.Mecl_v:
            IMF_v_list = []
            mw_IMF_v_list = []
            for Z in self.Z_massfrac_v:
                imf = SingleStellarIMF(M, Z, self.SFR)
                IMF_v_list.append(imf.IMF_func(self.mstar_v))
                mw_IMF_v_list.append(imf.IMF_mass_weighted_func(self.mstar_v))
            IMF_Z_v_list.append(IMF_v_list)
            mw_IMF_Z_v_list.append(mw_IMF_v_list)
        return IMF_Z_v_list, mw_IMF_Z_v_list
    
    def sIMF_subplot(self):
        return self.plots.sIMF_subplot(self.metallicity_v, self.Mecl_v, 
                                    self.mstar_v, self.IMF_Z_v_list)
        
    def mw_sIMF_subplot(self):
        return self.plots.mw_sIMF_subplot(self.metallicity_v, self.Mecl_v, 
                                    self.mstar_v, self.mw_IMF_Z_v_list)
    
    def sIMF_subplot_Mecl(self):
        return self.plots.sIMF_subplot_Mecl(self.metallicity_v, self.Mecl_v,
                                            self.mstar_v, self.IMF_Z_v_list)

    def sIMF_subplot_SFR(self):
        return self.plots.sIMF_subplot_SFR(self.metallicity_v, self.SFR_v,
                                            self.mstar_v, self.IMF_Z_v_list)

    

class InstanceIGIMF(Vectors):
    def __init__(self, metal_mass_fraction:float, SFR:float, computeV=False):
        super().__init__()
        self.o_IGIMF = inst.IGIMF(metal_mass_fraction, SFR)
        self.__dict__.update(self.o_IGIMF.__dict__)
        if computeV is True:
            self.IGIMF_v = self.o_IGIMF.IGIMF_func(self.mstar_v)


class IGIMFGrid(InstanceIGIMF):   
    '''Creates IGIMF grids'''  
    def __init__(self, metal_mass_fraction=0.1*0.0134, SFR=1):
        super().__init__(metal_mass_fraction, SFR)
        
    def by_Z_by_SFR(self):
        IGIMF_v_list = []
        for S in self.SFR_v:
            IGIMF_list = []
            for Z in self.Z_massfrac_v:
                igimf = InstanceIGIMF(Z, S, computeV=True)
                #print(igimf.IGIMF_v)
                IGIMF_list.append(igimf.IGIMF_v)
            IGIMF_v_list.append(IGIMF_list)
        return IMF_Z_S_v_list 
    
    def by_Z_by_SFR_pickle(self):
        import pickle
        #df = {'SFR':[], 'metal_mass_fraction':[], 'mass_star':[], 'IGIMF':[]}
        for S in self.SFR_v:
            IGIMF_list = []
            for Z in self.Z_massfrac_v:
                igimf = InstanceIGIMF(Z, S, computeV=True)
                #df['SFR'].append(S)
                #df['metal_mass_fraction'].append(Z)
                #df['mass_star'].append(self.mstar_v)
                #df['IGIMF'].append(igimf.IGIMF_func(self.mstar_v))
                df = {}
                df['SFR'] = S
                df['metal_mass_fraction'] = Z
                df['mass_star'] = self.mstar_v
                df['IGIMF'] = igimf.IGIMF_v
                df = pd.DataFrame(df)
                pickle.dump(df,open(f'grid/igimf_SFR{S}_Z{Z}.pkl','wb'))
        return None


class IGIMFplots(InstanceIGIMF):
    def __init__(self, metal_mass_fraction=0.1*0.0134, SFR=1):
        super().__init__(metal_mass_fraction, SFR)
        import glob
        txts = glob.glob('grid/resolution15/*pkl')
        #txts.remove('.DS_Store')
        #txts.remove('resolution50')
        df = pd.DataFrame({col:[] for col in ['SFR', 'metal_mass_fraction',
                                            'mass_star', 'IGIMF']})
        
        print('Importing pickle files')
        for txt in txts:
            df_txt = pickle.load(open(txt, 'rb'))
            df = pd.concat([df,df_txt])
        print('Pickle files imported')
        #df_grids = copy(df)
        self.df = df

        SFR_v = np.unique(df['SFR'])
        metal_mass_fraction_v = np.unique(df['metal_mass_fraction'])
        mstar_v = np.unique(df['mass_star'])
        self.mstar_v = mstar_v
        self.metal_mass_fraction_v = metal_mass_fraction_v
        self.SFR_v = SFR_v

        self.SFR_v_fit = np.logspace(np.log10(np.min(SFR_v)), np.log10(np.max(SFR_v)))
        self.mstar_v_fit = np.logspace(np.log10(np.min(mstar_v)), np.log10(np.max(mstar_v)))
        self.IGIMF_vmetal_mass_fraction_v_fit = np.logspace(np.log10(np.min(metal_mass_fraction_v)), np.log10(np.max(metal_mass_fraction_v)))

        
    def sIMF_subplot(self):
        return self.plots.sIMF_subplot(self.metallicity_v, self.Mecl_v, 
                                    self.mstar_v, self.IMF_Z_v_list)
        
    def mw_sIMF_subplot(self):
        return self.plots.mw_sIMF_subplot(self.metallicity_v, self.Mecl_v, 
                                    self.mstar_v, self.mw_IMF_Z_v_list)
    
    def sIMF_subplot_Mecl(self):
        return self.plots.sIMF_subplot_Mecl(self.metallicity_v, self.Mecl_v,
                                            self.mstar_v, self.IMF_Z_v_list)

    def sIMF_subplot_SFR(self):
        return self.plots.sIMF_subplot_SFR(self.metallicity_v, self.SFR_v,
                                            self.mstar_v, self.IMF_Z_v_list)




class Alpha3_grid:
    def __init__(self):
        from sklearn.model_selection import ParameterGrid

        

if __name__ == '__main__':
    metal_mass_fraction = 1e-1 * 0.0134
    M_igal = 1e10
    M_ecl = 1e5
    print('Creating the SingleECMF object')
    o_igimf = SingleECMF(M_igal=M_igal, metal_mass_fraction=metal_mass_fraction, downsizing_bool=True)
    print('Created the SingleECMF object')
    o_igimf.plots.Cook23_plot()
    print('Created Cook23 plot')
    o_igimf.ECMF_plot()
    print('Created ECMF_plot')
    
    o_vector = Vectors(resolution=30)
    ecmf_by_SFR = ECMFbySFR(SFR=o_vector.SFR_v[10], metal_mass_fraction=metal_mass_fraction)
    ecmf_by_SFR.all_plots()
    
    stellar_IMF = SingleStellarIMF(M_ecl, metal_mass_fraction, o_igimf.SFR)
    stellar_IMF.IMF_plot()
    
    SNCC_lowmass = SNCC_vs_lowmass(stellar_IMF)
    
    sIMF_by_Mtot = StellarIMFbyMtot(metal_mass_fraction, o_igimf.SFR)
    sIMF_by_Mtot.IMF_plots()
    
    sIMF_by = StellarmmaxMecl(metal_mass_fraction, o_igimf.SFR)
    sIMF_by.k_Z_plot()
    
    sIMF_by_Z = StellarIMFbyZbyMecl(o_igimf.SFR)
    sIMF_by_Z.sIMF_subplot()
    sIMF_by_Z.mw_sIMF_subplot()
    sIMF_by_Z.sIMF_subplot_Mecl()
    sIMF_by_Z.sIMF_subplot_SFR()
    
    
    #instance_IGIMF = InstanceIGIMF(metal_mass_fraction, o_igimf.SFR,
    #                               computeV=True)
    #print(instance_IGIMF.IGIMF_v)
    
    #create_grid = IGIMFGrid()
    #create_grid.by_Z_by_SFR_pickle()