from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import optimize 
import numpy as np
import pandas as pd
from . import  pvaps
from .root_functions import vfall, vfall_find_root, find_rg, moment, solve_force_balance
import time
from . import justdoit as jdi

def direct_solver(temperature, pressure, condensibles, gas_mw, gas_mmr, rho_p , mw_atmos, 
                        gravity, kzz, fsed, mh, sig, rmin, nrad, d_molecule,eps_k,c_p_factor,
                        tol = 1e-15, refine_TP = True,og_vfall=True, analytical_rg = True):
    """
    Given an atmosphere and condensates, calculate size and concentration
    of condensates in balance between eddy diffusion and sedimentation.

    Parameters
    ----------
    temperature : ndarray
        Temperature at each layer (K)
    pressure : ndarray
        Pressure at each layer (dyn/cm^2)
    condensibles : ndarray or list of str
        List or array of condensible gas names
    gas_mw : ndarray
        Array of gas mean molecular weight from `gas_properties`
    gas_mmr : ndarray 
        Array of gas mixing ratio from `gas_properties`
    rho_p : float 
        density of condensed vapor (g/cm^3)
    mw_atmos : float 
        Mean molecular weight of the atmosphere
    gravity : float 
        Gravity of planet cgs
    kz : float or ndarray
        Kzz in cgs, either float or ndarray depending of whether or not 
        it is set as input
    fsed : float 
        Sedimentation efficiency, unitless
    mh : float 
        Atmospheric metallicity in NON log units (e.g. 1 for 1x solar)
    sig : float 
        Width of the log normal particle distribution
    d_molecule : float 
        diameter of atmospheric molecule (cm) (Rosner, 2000)
        (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        Set in Atmosphere constants 
    eps_k : float 
        Depth of the Lennard-Jones potential well for the atmosphere 
        Used in the viscocity calculation (units are K) (Rosner, 2000)
    c_p_factor : float 
        specific heat of atmosphere (erg/K/g) . Usually 7/2 for ideal gas
        diatomic molecules (e.g. H2, N2). Technically does slowly rise with 
        increasing temperature
    tol : float 
        Tolerance for direct solver
    refine_TP : bool
        Option to refine temperature-pressure profile for direct solver 
    analytical_rg : bool
        Option to use analytical expression for rg, or alternatively deduce rg from calculation
        Calculation option will be most useful for future inclusions of alternative particle size distributions

    Returns
    -------
    qc : ndarray 
        condenstate mixing ratio (g/g)
    qt : ndarray 
        gas + condensate mixing ratio (g/g)
    rg : ndarray
        geometric mean radius of condensate  (cm) 
    reff : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz : ndarray 
        number column density of condensate (cm^-3)
    qc_path : ndarray 
        vertical path of condensate 
    pres : ndarray
        Pressure at each layer (dyn/cm^2)
    temp : ndarray
        Temperature at each layer (K)
    z : ndarray
        altitude of each layer (cm)
    
    """

    ngas =  len(condensibles)
    # refine temperature-pressure profile
    # this improves the accuracy of the mmr calculation
    (z, pres, P_z, temp, T_z, T_P, kz) = generate_altitude(pressure, temperature, kzz, gravity, 
                                                                mw_atmos, refine_TP) 

    pres_out = pressure
    temp_out = temperature
    z_out = interp1d(pres, z)(pres_out)

    mixl_out = np.zeros((len(pres_out), ngas))
    qc_out = np.zeros((len(pres_out), ngas))
    qt_out = np.zeros((len(pres_out), ngas))
    rg_out = np.zeros((len(pres_out), ngas))
    reff_out = np.zeros((len(pres_out), ngas))
    ndz_out = np.zeros((len(pres_out), ngas))
    qc_path = np.zeros(ngas)

    # find mmr and particle distribution for every condensible
    # perform calculation on refined TP profile but output values corresponding to initial profile
    for i, igas in zip(range(ngas), condensibles):
        gas_name = igas
        qc, qt, rg, reff, ndz, dz, qc_path[i], mixl = calc_qc(z, P_z, T_z, T_P, kz,
            gravity, gas_name, gas_mw[i], gas_mmr[i], rho_p[i], mw_atmos, mh, fsed, sig, rmin, nrad, 
            d_molecule,eps_k,c_p_factor,
            tol,og_vfall, analytical_rg)

        # generate qc values for original pressure data
        qc_out[:,i] = interp1d(pres, qc)(pres_out)
        qt_out[:,i] = interp1d(pres, qt)(pres_out)
        rg_out[:,i] = interp1d(pres, rg)(pres_out)
        reff_out[:,i] = interp1d(pres, reff)(pres_out)
        mixl_out[:,i] = interp1d(pres, mixl)(pres_out)

        ndz_temp = ndz/dz
        dz_new = np.insert(-(z_out[1:]-z_out[:-1]), len(z_out)-1, 1e-8)
        ndz_out[:,i] = interp1d(pres, ndz_temp)(pres_out) * dz_new

    return (qc_out, qt_out, rg_out, reff_out, ndz_out, qc_path, pres_out, temp_out, z_out,mixl_out)

def calc_qc(z, P_z, T_z, T_P, kz, gravity, gas_name, gas_mw, gas_mmr, rho_p, mw_atmos, 
                    mh, fsed, sig, rmin, nrad, d_molecule,eps_k,c_p_factor,
                    tol, og_vfall=True, analytical_rg=True, supsat=0):
    """
    Calculate condensate optical depth and effective radius for atmosphere,
    assuming geometric scatterers. 

    z : float 
        Altitude  cm 
    P_z: function
        Pressure at altitude z (dyne/cm^2)
    T_z: function
        Temperature at altitude z (K)
    T_P: function
        Temperature at pressure P (K)
    kz : float or ndarray
        Kzz in cgs
    gravity : float 
        Gravity of planet cgs 
    gas_name : str 
        Name of condensate 
    gas_mw : ndarray
        Array of gas mean molecular weight from `gas_properties`
    gas_mmr : ndarray 
        Array of gas mixing ratio from `gas_properties`
    rho_p : float 
        density of condensed vapor (g/cm^3)
    mw_atmos : float 
        Mean molecular weight of the atmosphere
    mh : float 
        Metallicity NON log solar (1 = 1x solar)
    fsed : float 
        Sedimentation efficiency (unitless)
    sig : float 
        Width of the log normal particle distrubtion 
    d_molecule : float 
        diameter of atmospheric molecule (cm) (Rosner, 2000)
        (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        Set in Atmosphere constants 
    eps_k : float 
        Depth of the Lennard-Jones potential well for the atmosphere 
        Used in the viscocity calculation (units are K) (Rosner, 2000)
    c_p_factor : float 
        specific heat of atmosphere (erg/K/g) . Usually 7/2 for ideal gas
        diatomic molecules (e.g. H2, N2). Technically does slowly rise with 
        increasing temperature
    tol : float 
        Tolerance for direct solver
    analytical_rg : bool
        Option to use analytical expression for rg, or alternatively deduce rg from calculation
        Calculation option will be most useful for future inclusions of alternative particle size distributions
    supsat : float, optional
        Default = 0 , Saturation factor (after condensation)

    Returns
    -------
    qc_out : ndarray 
        condenstate mixing ratio (g/g)
    qt_out : ndarray 
        gas + condensate mixing ratio (g/g)
    rg : ndarray
        geometric mean radius of condensate  (cm) 
    reff : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz : ndarray 
        number column density of condensate (cm^-3)
    qc_path : ndarray 
        vertical path of condensate 
    """
    #   universal gas constant (erg/mol/K)
    R_GAS = 8.3143e7
    AVOGADRO = 6.02e23
    K_BOLTZ = R_GAS / AVOGADRO
    PI = np.pi 
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
        return dTdlnP(P) / ( T / c_p_factor )
    #   convective mixing length scale (cm) 
    def mixl(T, P):
        return np.max( [0.10, lapse_ratio(T, P)]) * scale_h(T)
    get_pvap = getattr(pvaps, gas_name)
    def pvap(T, P):
        if gas_name == 'Mg2SiO4':
            return get_pvap(float(T), float(P), mh=mh)
        else:
            return get_pvap(float(T), mh=mh)
    #   mass mixing ratio of saturated layer
    def qvs(T, P):
        qvs_val =  (supsat + 1) * pvap(T, P) / (r_cloud * T) / rho_atmos(T, P) 
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
    z = z[::-1]
    kz = kz[::-1]
    #   define ode
    def AM4(z, q):
        P = P_z(z); T = T_z(z); 
        qc_val = max([0., q - qvs(T, P)])
        return - fsed *  qc_val / mixl(T, P)

    sol = solve_ivp(lambda t, y: AM4(t, y), [z[0], z[len(z)-1]], [q_below], method = "RK23", 
            rtol = 1e-12, atol = tol, dense_output=True, t_eval=z)
    qt = sol.sol

    mixl_out = np.zeros(len(z))
    qc_out = np.zeros(len(z))
    qt_out = np.zeros(len(z))
    p_out = np.zeros(len(z))
    for i in range(len(z)):
        p_out[i] = P_z(z[i])
        qt_out[i] = qt(z[i])
        T = T_z(z[i]); P = P_z(z[i])
        qc_out[i] = max([0., qt_out[i] - qvs(T, P)])
        mixl_out[i] =  mixl(T, P)
    #   --------------------------------------------------------------------
    #   Find <rw> corresponding to <w_convect> using function vfall()

    #   precision of vfall solution (cm/s)
    dz = np.insert(z[1:]-z[:-1], 0, 1e-8)
    rw = np.zeros(len(qc_out))
    rg = np.zeros(len(qc_out))
    reff = np.zeros(len(qc_out))
    ndz = np.zeros(len(qc_out))
    qc_path = 0.
    for i in range(len(qc_out)):

        if qc_out[i] == 0.0: # layer is cloud free
            rg[i] = 0.; reff[i] = 0; ndz[i] = 0.

        else:
            #   range of particle radii to search (cm)
            rlo = 1.e-10
            rhi = 10.
            find_root = True
            while find_root:
                try:
                    P = p_out[i]; T = T_P(p_out[i]); k = kz[i]
                    if og_vfall:
                        rw_temp = optimize.root_scalar(vfall_find_root, bracket=[rlo, rhi], method='brentq', 
                            args=(gravity, mw_atmos, mfp(T, P),  visc(T), T, P, rho_p, w_convect(T, P, k)))
                    else:
                        rw_temp = solve_force_balance("rw", w_convect(T, P, k), gravity, mw_atmos, mfp(T, P),
                                                    visc(T), T, P, rho_p, rlo, rhi)
                    find_root = False
                except ValueError:
                    rlo = rlo/10
                    rhi = rhi*10

            #fall velocity particle radius 
            if og_vfall: rw[i] = rw_temp.root
            else: rw[i] = rw_temp
    
            #   geometric std dev of lognormal size distribution ** sig is the geometric std dev
            lnsig2 = 0.5*np.log( sig )**2
            #   sigma floor for the purpose of alpha calculation
            sig_alpha = np.max( [1.1, sig] )    

            #if fsed > 1 :
            #    #   Bulk of precip at r > rw: exponent between rw and rw*sig
            #    r_ = rw[i]*sig_alpha
            #else:
            #    #   Bulk of precip at r < rw: exponent between rw/sig and rw
            #    r_ = rw[i]/sig_alpha

            #if og_vfall:
            #    vf = vfall(r_, gravity, mw_atmos, mfp(T,P), visc(T), T, P,  rho_p)
            #else:
            #    vlo = 1e0; vhi = 1e6
            #    vf = solve_force_balance("vfall", r_, gravity, mw_atmos, mfp(T, P),
            #                                        visc(T), T, P, rho_p, vlo, vhi)

            #alpha = (np.log( vf / w_convect(T, P, k) )
            #                     / np.log( r_ / rw[i] ))

            #   find alpha for power law fit vf = w(r/rw)^alpha
            def pow_law(r, alpha):
                return np.log(w_convect(T, P, k)) + alpha * np.log (r / rw[i]) 

            r_, rup, dr = jdi.get_r_grid(r_min = rmin, n_radii = nrad)
            vfall_temp = []
            for j in range(len(r_)):
                if og_vfall:
                    vfall_temp.append(vfall(r_[j], gravity, mw_atmos, mfp(T, P), visc(T), T, P, rho_p))
                else:
                    vlo = 1e0; vhi = 1e6
                    find_root = True
                    while find_root:
                        try:
                            vfall_temp.append(solve_force_balance("vfall", r_[j], gravity, mw_atmos, 
                                mfp(T, P), visc(T), T, P, rho_p, vlo, vhi))
                            find_root = False
                        except ValueError:
                            vlo = vlo/10
                            vhi = vhi*10

            pars, cov = optimize.curve_fit(f=pow_law, xdata=r_, ydata=np.log(vfall_temp), p0=[0], 
                                bounds=(-np.inf, np.inf))
            alpha = pars[0]

            if analytical_rg:
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

            else:
                #   range of particle radii to search (cm)
                rlo = 1.e-10
                rhi = 1.e2
                #   geometric mean radius of size distribution
                rg_temp = optimize.root_scalar(find_rg, bracket=[rlo, rhi], method='brentq', 
                                                    args=(fsed, rw[i], alpha, np.log(sig)))
                rg[i] = rg_temp.root

                #   droplet effective radius (cm)
                #   ratio of third to second moment of size distribution
                reff[i] = moment(3, np.log(sig), 0., rg[i]) / moment(2, np.log(sig), 0., rg[i])

                #   column droplet number concentration (cm^-2)
                ndz[i] = (3 * fsed * rw[i]**alpha * qc_out[i] * rho_atmos(T, P) * dz[i] / 
                            (4 * np.pi * rho_p * moment(3+alpha, np.log(sig), 0., rg[i])))


        if i > 0:   
            qc_path = (qc_path + qc_out[i-1] *
                            ( p_out[i-1] - p_out[i] ) / gravity)

    return (qc_out[::-1], qt_out[::-1], rg[::-1], reff[::-1], ndz[::-1], dz[::-1], qc_path, mixl_out[::-1])

def generate_altitude(pres, temp, kz, gravity, mw_atmos, refine_TP=True, eps=10):
    """
    Refine temperature pressure profile according to maximum temperature-difference
    between pressure layers.

    pres : ndarray
        Pressure at each layer (dyn/cm^2)
    temp : ndarray
        Temperature at each layer (K)
    kz : float or ndarray
        Kzz in cgs
    gravity : float 
        Gravity of planet cgs 
    mw_atmos : float 
        Mean molecular weight of the atmosphere
    refine_TP : bool
        Option to refine temperature-pressure profile for direct solver 
    eps : float
        maximum temperature difference between pressure layers

    Returns
    -------
    z : float 
        Altitude  cm 
    pres_ : ndarray
        Pressure at each layer (dyn/cm^2)
    P_z: function
        Pressure at altitude z (dyne/cm^2)
    temp_ : ndarray
        Temperature at each layer (K)
    T_z: function
        Temperature at altitude z (K)
    T_P: function
        Temperature at pressure P (K)
    kz_ : float or ndarray
        Kzz in cgs
    """
    #   universal gas constant (erg/mol/K)
    R_GAS = 8.3143e7
    #   specific gas constant for atmosphere (erg/K/g)
    r_atmos = R_GAS / mw_atmos
    #   atmospheric scale height (cm) 
    def H(T): 
        return r_atmos * T / gravity

    T_P = UnivariateSpline(pres, temp)
    # this is grim fix this
    if len(pres) == len(kz):
        kz_P = interp1d(pres, kz) 
    else:
        kz_P = interp1d(pres, kz[:-1]) 

    pres_ = pres
    if refine_TP:
        #   we use barometric formula which assumes constant temperature 
        #   define maximum difference between temperature values which if exceeded, reduce pressure stepsize
        n = len(pres_)
        while max(abs(T_P(pres_[1:]) - T_P(pres_[:-1]))) > eps:
            indx = np.where(abs(T_P(pres_[1:]) - T_P(pres_[:-1])) > eps)[0]
            mids = pres_[indx] + (pres_[indx+1] - pres_[indx]) / 2
            pres_ = np.insert(pres_, indx+1, mids)
    pres_ = pres_[::-1]
    
    z = np.zeros(len(pres_))
    T = np.zeros(len(pres_)); T[0] = temp[len(temp)-1]
    K = np.zeros(len(pres_)); K[0] = kz[len(kz)-1]
    for i in range(len(pres_) - 1):
        T[i+1] = T_P(pres_[i+1])
        K[i+1] = kz_P(pres_[i+1])
        dz = - H(T[i+1]) * np.log(pres_[i+1] / pres_[i]) 
        z[i+1] = z[i] + dz
            
    P_z = UnivariateSpline(z, pres_)
    T_z = UnivariateSpline(z, T)
    temp_ = T
    kz_ = K
    
    return (z[::-1], pres_[::-1], P_z, temp_[::-1], T_z, T_P, kz_[::-1])
    #return (z, pres_, P_z, temp_, T_z, T_P, kz_)
