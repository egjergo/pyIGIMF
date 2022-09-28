import IGIMF3
import numpy as np
import pandas as pd

resolution = 20

Zsun = 0.0142
delta_alpha=63
delta_t = 1e7
Mecl_min = 5
Mecl_max = 1e9
m_max = 150
m_min = 0.08

M_igal_v = np.logspace(6, 11,num=resolution)
M_ecl_v = np.logspace(np.log10(Mecl_min), np.log10(Mecl_max), num=resolution)
time_v = np.logspace(-3, 0.5, num=resolution)
Z_rel = lambda t: (-np.reciprocal(5*t) -1.5)
Z_rel_v = np.array([Z_rel(t) for t in time_v])
Z_v = np.power(10, Z_rel_v)
Z_sqr_v = np.log10(np.divide(Z_v, Zsun))

def Z_plot(time_v, Z_v):
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.semilogy(time_v, Z_v, linewidth=3, color='magenta')
    ax.set_ylabel(r'$Z=M_Z/M_{gas}$', fontsize=15)
    ax.set_xlabel(f'Time [Gyr]', fontsize=15)
    ax.axhline(Zsun, color='black', linewidth=2)
    #plt.title(f'{M = :.2e} {Msun}', fontsize=15)
    ax.set_ylim(1e-8,1)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.tick_params(width=2)
    fig.tight_layout()
    #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
    plt.show(block=False)
    return None
Z_plot(time_v, Z_v)

def rho_cl(M_ecl):
    return 10**(0.61 * np.log10(M_ecl) + 2.85)
rho_cl_v = np.array([rho_cl(M) for M in M_ecl_v]) 

def rho_plot(M_ecl_v, rho_cl_v):
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    Msun = r'$M_{\odot}$'
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.loglog(M_ecl_v, rho_cl_v, linewidth=3, color='grey')
    ax.set_ylabel(r'$\rho_{cl}$', fontsize=15)
    ax.set_xlabel(r'$M_{ecl}$ '+f'[{Msun}]', fontsize=15)
    #plt.title(f'{M = :.2e} {Msun}', fontsize=15)
    #ax.set_ylim(1e-8,1)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.tick_params(width=2)
    fig.tight_layout()
    #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
    plt.show(block=False)
    return None
rho_plot(M_ecl_v, rho_cl_v)

def x_alpha_3_func(Z, rho_cl):
    return (-0.14 * Z + 0.99 * np.log10(rho_cl/1e6))
x_func = np.vectorize(x_alpha_3_func)
Z_mesh, rho_mesh = np.meshgrid(Z_sqr_v, rho_cl_v)
x_mesh = x_func(Z_mesh, rho_mesh)

def x_plot():
    from matplotlib import pyplot as plt
    from mpl_toolkits import mplot3d
    import matplotlib.ticker as ticker
    fig = plt.figure(figsize=(7,5))
    ax = plt.axes(projection='3d')
    #ax.contour3D(Z_mesh, rho_mesh, x_mesh, 50, cmap='plasma')
    ax.plot_surface(np.log10(Z_mesh), np.log10(rho_mesh), x_mesh, cmap='plasma')
    #ax.plot_surface(Z_sqr_v, rho_cl_v, x_mesh, 50, cmap='plasma')
    ax.set_xlabel('[Z]', fontsize=15, labelpad=15)
    ax.set_ylabel(r'$\log_{10}(\rho_{cl})$', fontsize=15, labelpad=15)
    ax.set_zlabel(r'x$_{\alpha_3}$', fontsize=15, labelpad=15)
    ax.tick_params(labelsize=15)
    ax.set_zlim(-4,5)
    plt.show(block=False)
x_plot()

def alpha_3_func(x_alpha_3):
    if x_alpha_3 < -0.87:
        return 2.3
    else:
        return -0.41 * x_alpha_3 + 1.94
xm_s1, xm_s2 = x_mesh.shape
alpha_3_mesh = np.reshape([[alpha_3_func(x_mesh[i,j]) for i in range(xm_s1)] for j in range(xm_s2)], x_mesh.shape)

def alpha_3_plot():
    from matplotlib import pyplot as plt
    from mpl_toolkits import mplot3d
    import matplotlib.ticker as ticker
    fig = plt.figure(figsize=(7,5))
    ax = plt.axes(projection='3d')
    #ax.contour3D(Z_mesh, rho_mesh, x_mesh, 50, cmap='plasma')
    ax.plot_surface(np.log10(Z_mesh), np.log10(rho_mesh), alpha_3_mesh, cmap='plasma_r')
    #ax.plot_surface(Z_sqr_v, rho_cl_v, x_mesh, 50, cmap='plasma')
    ax.set_xlabel('[Z]', fontsize=15, labelpad=15)
    ax.set_ylabel(r'$\log_{10}(\rho_{cl})$', fontsize=15, labelpad=15)
    ax.set_zlabel(r'${\alpha_3}$', fontsize=15, labelpad=15)
    #ax.set_zlim(-4,5)
    ax.tick_params(labelsize=15)
    plt.show(block=False)
alpha_3_plot()

def alpha_1_func(metal_mass_fraction):
    return 1.3 + delta_alpha * (metal_mass_fraction - Zsun) 
def alpha_2_func(alpha_1):
    return 1 + alpha_1
alpha_1 = np.array([alpha_1_func(Z) for Z in Z_v])
alpha_2 = np.array([alpha_2_func(alpha1) for alpha1 in alpha_1])

def alpha12_plot(Z_v, alpha_1, alpha_2):
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    Msun = r'$M_{\odot}$'
    fig, ax1 = plt.subplots(1,1, figsize=(7,5))
    ax0 = ax1.twinx()
    ax0.plot(Z_v, alpha_1, linewidth=1, color='tab:red')
    ax0.set_ylabel(r'$\alpha_1$', fontsize=15, color='tab:red')
    ax0.set_xlabel(r'$Z=M_Z/M_{gas}$', fontsize=15)
    ax1.plot(Z_v, alpha_2, linewidth=2, color='tab:blue')
    ax1.set_ylabel(r'$\alpha_2$', fontsize=15, color='tab:blue')
    ax1.set_xlabel(r'$Z=M_Z/M_{gas}$', fontsize=15)
    #ax.set_ylim(1e-8,1)
    ax0.tick_params(width=2, axis='both', labelsize=15)
    ax1.tick_params(width=2, axis='both', labelsize=15)
    fig.tight_layout()
    #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
    plt.show(block=False)
    return None
alpha12_plot(Z_v, alpha_1, alpha_2)

def delta_tau(M_igal):
    return 8.16 * np.e**(-0.556 * np.log10(M_igal) + 3.401) + 0.027
def SFR_func(M_igal):
    downsizing_time = delta_tau(M_igal)
    return np.divide(M_igal, downsizing_time * 1e9), downsizing_time
val = np.array([SFR_func(Migal) for Migal in M_igal_v])
SFR = val[:,0]
downsizing_time = val[:,1]

def Migal_plot(M_igal_v, SFR, downsizing_time):
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    Msun = r'$M_{\odot}$'
    fig, ax1 = plt.subplots(1,1, figsize=(7,5))
    ax0 = ax1.twinx()
    ax0.loglog(M_igal_v, SFR, linewidth=3, color='tab:red')
    ax0.set_ylabel(f'SFR [{Msun}/yr]', fontsize=15, color='tab:red')
    ax0.set_xlabel(r'$M_{igal}$ '+f'[{Msun}]', fontsize=15)
    ax1.semilogx(M_igal_v, downsizing_time, linewidth=3, color='tab:blue')
    ax1.set_ylabel(r'$\Delta\tau$ [Gyr]', fontsize=15, color='tab:blue')
    ax1.set_xlabel(r'$M_{igal}$ '+f'[{Msun}]', fontsize=15)
    #ax.set_ylim(1e-8,1)
    ax0.tick_params(width=2, axis='both', labelsize=15)
    ax1.tick_params(width=2, axis='both', labelsize=15)
    fig.tight_layout()
    #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
    plt.show(block=False)
    return None
Migal_plot(M_igal_v, SFR, downsizing_time)

def beta_func(SFR_in_delta_t):
    return -0.106 * np.log10(SFR_in_delta_t) + 2
beta_ECMF = [beta_func(s) for s in SFR]

def beta_plot(SFR, beta_ECMF):
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    Msun = r'$M_{\odot}$'
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.semilogx(SFR, beta_ECMF, linewidth=3, color='purple')
    ax.set_ylabel(r'$\beta$', fontsize=15)
    ax.set_xlabel(f'SFR [{Msun}/yr]', fontsize=15)
    #plt.title(f'{M = :.2e} {Msun}', fontsize=15)
    #ax.set_ylim(1e-8,1)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.tick_params(width=2)
    fig.tight_layout()
    #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
    plt.show(block=False)
    return None
beta_plot(SFR, beta_ECMF)

igimf3 = IGIMF3.IGIMF(0,0)#, downsizing_time, t)

def embedded_cluster_mass_function(M_ecl, SFR_delta_t, M_max=None):
    r"""Eq. (8)"""
    if M_ecl>=Mecl_min:
        return igimf3.normalized(M_ecl, M_ecl**(-beta_func(SFR_delta_t)), condition=M_max)
    else:
        return 0.
        
def ECMF(SFR_delta_t):
    '''duplicate of stellar_IMF !!!!!!! '''
    k_ecl, M_max = igimf3.normalization(embedded_cluster_mass_function,
                                      SFR_delta_t * delta_t, Mecl_min, Mecl_max, SFR_delta_t,)
    ECMF_func = lambda M_ecl: M_ecl * k_ecl * embedded_cluster_mass_function(M_ecl, SFR_delta_t, M_max=M_max)
    ECMF_weighted_func = lambda M_ecl: igimf3.weighted_func(M_ecl, ECMF_func)
    return k_ecl, M_max, ECMF_func, ECMF_weighted_func

ECMF_val = [ECMF(S) for S in SFR]
ECMF_functions = []
for i,val in enumerate(ECMF_val):
    ECMF_functions.append(val[2])
    
ECMF_mesh = []
for ECMF_f in ECMF_functions:
    ECMF_mesh.append([ECMF_f(M) for M in M_ecl_v])
    
SFR_mesh, M_ecl_v_mesh = np.meshgrid(SFR, M_ecl_v)
ECMF_mesh_reshape = np.reshape(ECMF_mesh, (len(ECMF_functions), len(M_ecl_v)))
    
def ECMF_3D_plot():
    from matplotlib import pyplot as plt
    from mpl_toolkits import mplot3d
    import matplotlib.ticker as ticker
    Msun = r'$M_{\odot}$'
    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(np.log10(SFR_mesh), np.log10(M_ecl_v_mesh), np.log10(ECMF_mesh_reshape), cmap='magma')
    #ax.plot_surface(Z_sqr_v, rho_cl_v, x_mesh, 50, cmap='plasma')
    ax.set_xlabel(r'$M_{\rm ecl}$ [$\log_{10}(M_{\odot})$]', fontsize=15, labelpad=15)
    ax.set_ylabel(f'SFR [{Msun}/yr]', fontsize=15, labelpad=15)
    ax.set_zlabel(r'$\xi_{ECMF}$', fontsize=15, labelpad=15)
    #ax.set_zlim(-4,5)
    ax.tick_params(labelsize=15)
    plt.show(block=False)
ECMF_3D_plot()

k_ecl1, M_max1, ECMF_func1, ECMF_weighted_func1 = ECMF(1e3)
ECMF_v = np.array([ECMF_func1(M) for M in M_ecl_v])
def ECMF_plot(M_ecl_v, ECMF_v):
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    Msun = r'$M_{\odot}$'
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.loglog(M_ecl_v, ECMF_v, linewidth=3, color='#01153E')
    ax.set_ylabel(r'$\xi_{ECMF}$', fontsize=15)
    ax.set_xlabel(r'E. cluster mass $M_{\rm ecl}$ [$\log_{10}(M_{\odot})$]', fontsize=15)
    #plt.title(f'{M = :.2e} {Msun}', fontsize=15)
    #ax.set_ylim(1e-8,1)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax.tick_params(width=2)
    fig.tight_layout()
    #plt.savefig(f'Z_plot_{name}.pdf', bbox_inches='tight')
    plt.show(block=False)
    return None
ECMF_plot(M_ecl_v, ECMF_v)