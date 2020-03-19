from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline 
from scipy import optimize 
import numpy as np
import pandas as pd
from . import  pvaps
from .root_functions import vfall, vfall_find_root

def direct_solver(temperature, pressure, condensibles, gas_mw, gas_mmr, rho_p , mw_atmos, 
                        gravity, kz, fsed, mh, sig, refine_TP = False):

    ngas =  len(condensibles)
    (z, pres, P_z, temp, T_z, T_P) = generate_altitude(pressure, temperature, gravity, mw_atmos, refine_TP) 

    qc = np.zeros((len(z), ngas))
    qt = np.zeros((len(z), ngas))
    rg = np.zeros((len(z), ngas))
    reff = np.zeros((len(z), ngas))
    ndz = np.zeros((len(z), ngas))
    qc_path = np.zeros((len(z), ngas))
    for i, igas in zip(range(ngas), condensibles):
        gas_name = igas
        qc[:,i], qt[:,i], rg[:,i], reff[:,i], ndz[:,i], qc_path[:,i] = calc_qc(z, P_z, T_z, T_P, kz,
            gravity, gas_name, gas_mw[i], gas_mmr[i], rho_p[i], mw_atmos, mh, fsed, sig)


    return (qc[::-1], qt[::-1], rg[::-1], reff[::-1], ndz[::-1], qc_path[::-1], 
                    pres[::-1], temp[::-1], z[::-1])

def calc_qc(z, P_z, T_z, T_P, kz, gravity, gas_name, gas_mw, gas_mmr, rho_p, mw_atmos, mh, fsed, sig):
    #   universal gas constant (erg/mol/K)
    R_GAS = 8.3143e7
    AVOGADRO = 6.02e23
    K_BOLTZ = R_GAS / AVOGADRO
    PI = np.pi 
    #   diameter of atmospheric molecule (cm) (Rosner, 2000)
    #   (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
    d_molecule = 2.827e-8
    #   Depth of the Lennard-Jones potential well for the atmosphere 
    # Used in the viscocity calculation (units are K) (Rosner, 2000)
    #   (78.6 for air, 71.4 for N2, 59.7 for H2)
    eps_k = 59.7
    #   specific gas constant for atmosphere (erg/K/g)
    r_atmos = R_GAS / mw_atmos
    #   specific gas constant for cloud (erg/K/g)
    r_cloud = R_GAS/ gas_mw
    #   atmospheric mean free path (cm)
    def mfp(T, P):
        #   atmospheric number density (molecules/cm^3)
        n_atmos = P / ( K_BOLTZ * T )
        return  1. / ( np.sqrt(2.) * n_atmos * PI * d_molecule**2 )
    #   atmospheric scale height (cm) 
    def scale_h(T): 
        return r_atmos * T / gravity
    #   atmospheric density (g/cm^3)
    def rho_atmos(T, P):
        return P / ( r_atmos * T )
    def dHdP(P):
        return r_atmos * dTdP(P) / gravity
    #   lapse ratio ** not sure what this physically is
    def lapse_ratio(T, P):
        dTdP =  T_P.derivative()
        def dTdlnP(P):
            return P * dTdP(P)
        return dTdlnP(P) / ( 2./7.* T )
    #   convective mixing length scale (cm) 
    def mixl(T, P):
        return np.max( [0.10, lapse_ratio(T, P)]) * scale_h(T)
    get_pvap = getattr(pvaps, gas_name) # fix to work for Mg2SiO4 
    def pvap(T):
        return get_pvap(float(T), mh=mh)
    #   super saturation factor
    supsat = 0.
    fs = supsat + 1
    #   mass mixing ratio of saturated layer
    def qvs(T, P):
        qvs_val =  fs * pvap(T) / (r_cloud * T) / rho_atmos(T, P) 
        return qvs_val
    # atmospheric viscosity (dyne s/cm^2)
    # EQN B2 in A & M 2001, originally from Rosner+2000
    # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems (Dover: Mineola)
    def visc(T):
        return (5./16.*np.sqrt( PI * K_BOLTZ * T * (mw_atmos/AVOGADRO)) /
        ( PI * d_molecule**2 ) /
        ( 1.22 * ( T / eps_k )**(-0.16) ))
    #   convective velocity scale (cm/s) from mixing length theory
    def w_convect(T, P, kz):
        return kz / mixl(T, P) 

    ##  Define and solve ODE (4) in AM2001 using scipy solve_ivp ---------------------------------------------
    q_below = gas_mmr
    #   define ode
    def mix_sed(z, q):
        P = P_z(z); T = T_z(z)
        qc_val = max([0., q - qvs(T, P)])
        return - fsed *  qc_val / mixl(T, P)

    sol = solve_ivp(lambda t, y: mix_sed(t, y), [z[0], z[len(z)-1]], [q_below], method = "RK23", 
            rtol = 1e-9, atol = 1e-9, dense_output=True, t_eval=z)
    z_vals = sol.t
    qt = sol.sol

    qc_out = np.zeros(len(z))
    qt_out = np.zeros(len(z))
    p_out = np.zeros(len(z))
    qvs_out = np.zeros(len(z))
    for i in range(len(z)):
        p_out[i] = P_z(z[i])
        qt_out[i] = qt(z[i])
        T = T_z(z[i]); P = P_z(z[i])
        qc_out[i] = max([0., qt_out[i] - qvs(T, P)])
        qvs_out[i] = qvs(T, P)

    #   --------------------------------------------------------------------
    #   Find <rw> corresponding to <w_convect> using function vfall()

    #   precision of vfall solution (cm/s)
    dz = np.insert(z[1:]-z[:-1], 0, 0)
    rw = np.zeros(len(qc_out))
    rg = np.zeros(len(qc_out))
    reff = np.zeros(len(qc_out))
    ndz = np.zeros(len(qc_out))
    for i in range(len(qc_out)):
        #   range of particle radii to search (cm)
        rlo = 1.e-10
        rhi = 10.
        find_root = True
        while find_root:
            try:
                P = p_out[i]; T = T_P(p_out[i]); k = kz[i]
                rw_temp = optimize.root_scalar(vfall_find_root, bracket=[rlo, rhi], method='brentq', 
                    args=(gravity, mw_atmos, mfp(T, P),  visc(T), T, P, rho_p, w_convect(T, P, k)))
                find_root = False
            except ValueError:
                rlo = rlo/10
                rhi = rhi*10

        #fall velocity particle radius 
        rw[i] = rw_temp.root
    
        #   geometric std dev of lognormal size distribution
        lnsig2 = 0.5*np.log( sig )**2
        #   sigma floor for the purpose of alpha calculation
        sig_alpha = np.max( [1.1, sig] )    

        if fsed > 1 :

            #   Bulk of precip at r > rw: exponent between rw and rw*sig
            alpha = (np.log(
                            vfall( rw[i]*sig_alpha, gravity, mw_atmos, mfp(T, P), visc(T), T, P, rho_p ))
                            / w_convect(T, P, k)
                                / np.log( sig_alpha ))

        else:
            #   Bulk of precip at r < rw: exponent between rw/sig and rw
            alpha = (np.log(
                            w_convect(T, P, k) / vfall( rw[i]/sig_alpha, gravity, mw_atmos, mfp(T, P),
                                visc(T), T, P, rho_p) )
                                    / np.log( sig_alpha ))

        #     EQN. 13 A&M 
        #   geometric mean radius of lognormal size distribution
        rg[i] = (fsed**(1./alpha) *
                    rw[i] * np.exp(-(alpha + 6) * lnsig2))

        #   droplet effective radius (cm)
        reff[i] = rg[i] * np.exp(5 * lnsig2)

        #      EQN. 14 A&M
        #   column droplet number concentration (cm^-2)
        ndz[i] = (3 * rho_atmos(T, P) * qc_out[i] * dz[i] /
                    ( 4 * np.pi * rho_p * rg[i]**3 ) * np.exp(-9 * lnsig2))

    #qc_path = (qc_path[i] + qc[iz,i]*
    #                            ( p_top[iz+1] - p_top[iz] ) / gravity)
    print('not calculating qc_path here')
    qc_path = np.zeros(len(qc_out))

    return (qc_out, qt_out, rg, reff, ndz, qc_path)

def generate_altitude(pres, temp, gravity, mw_atmos, refine_TP):  
    #   universal gas constant (erg/mol/K)
    R_GAS = 8.3143e7
    #   specific gas constant for atmosphere (erg/K/g)
    r_atmos = R_GAS / mw_atmos
    #   atmospheric scale height (cm) 
    def H(T): 
        return r_atmos * T / gravity

    T_P = UnivariateSpline(pres, temp)

    pres_ = pres[::-1]
    if refine_TP:
        #   we use barometric formula which assumes constant temperature 
        #   define maximum difference between temperature values which if exceeded, reduce pressure stepsize
        eps = 10 
        while max(abs(T_P(pres_[1:]) - T_P(pres_[:-1]))) > eps:
            print("warning in altitude calculation: temperature gradient exceeds set threshold: use smaller steps")
            n = n * 2
            print("setting n = ", n)
            pres_ = np.logspace(np.log10(pres[0]), np.log10(pres[len(pres)-1]), n)
            pres_ = pres_[::-1]
    
    
    z = np.zeros(len(pres_))
    T = np.zeros(len(pres_)); T[0] = temp[len(temp)-1]
    for i in range(len(pres_) - 1):
        T[i+1] = T_P(pres_[i+1])
        dz = - H(T[i+1]) * np.log(pres_[i+1] / pres_[i]) 
        z[i+1] = z[i] + dz
            
    P_z = UnivariateSpline(z, pres_)
    T_z = UnivariateSpline(z, T)
    temp_ = T_P(pres_)
    
    return (z, pres_, P_z, temp_, T_z, T_P)
