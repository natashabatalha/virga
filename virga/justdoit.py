import astropy.constants as c
import astropy.units as u
import pandas as pd
import numpy as np
import os
from scipy import optimize 
import PyMieScatt as ps

from .root_functions import advdiff, vfall,vfall_find_root,qvs_below_model, find_cond_t, solve_force_balance
from .calc_mie import fort_mie_calc, calc_new_mieff
from . import gas_properties
from . import pvaps
from .justplotit import plot_format, find_nearest_1d

from .direct_mmr_solver import direct_solver


def compute(atmo, directory = None, as_dict = True, og_solver = True, 
    direct_tol=1e-15, refine_TP = True, og_vfall=True, analytical_rg = True, do_virtual=True):

    """
    Top level program to run eddysed. Requires running `Atmosphere` class 
    before running this. 
    
    Parameters
    ----------
    atmo : class 
        `Atmosphere` class 
    directory : str, optional 
        Directory string that describes where refrind files are 
    as_dict : bool, optional 
        Default = False. Option to view full output as dictionary
    og_solver : bool, optional
        Default=True. BETA. Contact developers before changing to False.
         Option to change mmr solver (True = original eddysed, False = new direct solver)
    direct_tol : float , optional
        Only used if og_solver =False. Default = True. 
        Tolerance for direct solver
    refine_TP : bool, optional
        Only used if og_solver =False. 
        Option to refine temperature-pressure profile for direct solver 
    og_vfall : bool, optional
        Option to use original A&M or new Khan-Richardson method for finding vfall
    analytical_rg : bool, optional
        Only used if og_solver =False. 
        Option to use analytical expression for rg, or alternatively deduce rg from calculation
        Calculation option will be most useful for future 
        inclusions of alternative particle size distributions
    do_virtual : bool 
        If the user adds an upper bound pressure that is too low. There are cases where a cloud wants to 
        form off the grid towards higher pressures. This enables that. 

    Returns 
    -------
    opd, w0, g0
        Extinction per layer, single scattering abledo, asymmetry parameter, 
        All are ndarrays that are nlayer by nwave
    dict 
        Dictionary output that contains full output. See tutorials for explanation of all output.
    """
    mmw = atmo.mmw
    mh = atmo.mh
    condensibles = atmo.condensibles

    ngas = len(condensibles)

    gas_mw = np.zeros(ngas)
    gas_mmr = np.zeros(ngas)
    rho_p = np.zeros(ngas)

    # scale-height for fsed taken at Teff (default: temp at 1bar) 
    H = atmo.r_atmos * atmo.Teff / atmo.g
    
    
    #### First we need to either grab or compute Mie coefficients #### 
    for i, igas in zip(range(ngas),condensibles) : 

        #Get gas properties including gas mean molecular weight,
        #gas mixing ratio, and the density
        run_gas = getattr(gas_properties, igas)
        gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(mmw, mh=mh, gas_mmr=atmo.gas_mmr)

        #Get mie files that are already saved in 
        #directory
        #eventually we will replace this with nice database 
        qext_gas, qscat_gas, cos_qscat_gas, nwave, radius,wave_in = get_mie(igas,directory)

        if i==0: 
            nradii = len(radius)
            rmin = np.min(radius)
            radius, rup, dr = get_r_grid(rmin, n_radii=nradii)
            qext = np.zeros((nwave,nradii,ngas))
            qscat = np.zeros((nwave,nradii,ngas))
            cos_qscat = np.zeros((nwave,nradii,ngas))

        #add to master matrix that contains the per gas Mie stuff
        qext[:,:,i], qscat[:,:,i], cos_qscat[:,:,i] = qext_gas, qscat_gas, cos_qscat_gas

    #Next, calculate size and concentration
    #of condensates in balance between eddy diffusion and sedimentation

    #qc = condensate mixing ratio, qt = condensate+gas mr, rg = mean radius,
    #reff = droplet eff radius, ndz = column dens of condensate, 
    #qc_path = vertical path of condensate

    #   run original eddysed code
    if og_solver:
        if atmo.param is 'exp': 
            atmo.b = 6 * atmo.b * H # using constant scale-height in fsed
            fsed_in = (atmo.fsed-atmo.eps) 
        elif atmo.param is 'const':
            fsed_in = atmo.fsed
        qc, qt, rg, reff, ndz, qc_path, mixl, z_cld = eddysed(atmo.t_level, atmo.p_level, atmo.t_layer, atmo.p_layer, 
                                             condensibles, gas_mw, gas_mmr, rho_p , mmw, 
                                             atmo.g, atmo.kz, atmo.mixl, 
                                             fsed_in, atmo.b, atmo.eps, atmo.z_top, atmo.z_alpha, min(atmo.z), atmo.param,
                                             mh, atmo.sig, rmin, nradii,
                                             atmo.d_molecule,atmo.eps_k,atmo.c_p_factor,
                                             og_vfall, supsat=atmo.supsat,verbose=atmo.verbose,do_virtual=do_virtual)
        pres_out = atmo.p_layer
        temp_out = atmo.t_layer
        z_out = atmo.z

    
    #   run new, direct solver
    else:
        qc, qt, rg, reff, ndz, qc_path, pres_out, temp_out, z_out,mixl = direct_solver(atmo.t_layer, atmo.p_layer,
                                             condensibles, gas_mw, gas_mmr, rho_p , mmw, 
                                             atmo.g, atmo.kz, atmo.fsed, mh,atmo.sig, rmin, nradii, 
                                             atmo.d_molecule,atmo.eps_k,atmo.c_p_factor,
                                             direct_tol, refine_TP, og_vfall, analytical_rg)

            
    #Finally, calculate spectrally-resolved profiles of optical depth, single-scattering
    #albedo, and asymmetry parameter.    
    opd, w0, g0, opd_gas = calc_optics(nwave, qc, qt, rg, reff, ndz,radius,
                                       dr,qext, qscat,cos_qscat,atmo.sig, rmin, nradii)

    if as_dict:
        if atmo.param is 'exp':
            fsed_out = fsed_in * np.exp((atmo.z - atmo.z_alpha) / atmo.b ) + atmo.eps
        else: 
            fsed_out = fsed_in 
        return create_dict(qc, qt, rg, reff, ndz,opd, w0, g0, 
                           opd_gas,wave_in, pres_out, temp_out, condensibles,
                           mh,mmw, fsed_out, atmo.sig, nradii,rmin, z_out, atmo.dz_layer, 
                           mixl, atmo.kz, atmo.scale_h, z_cld) 
    else:
        return opd, w0, g0

def create_dict(qc, qt, rg, reff, ndz,opd, w0, g0, opd_gas,wave,pressure,temperature, gas_names,
    mh,mmw,fsed,sig,nrad,rmin,z, dz_layer, mixl, kz, scale_h, z_cld):
    return {
        "pressure":pressure/1e6, 
        "pressure_unit":'bar',
        "temperature":temperature,
        "temperature_unit":'kelvin',
        "wave":wave[:,0],
        "wave_unit":'micron',
        "condensate_mmr":qc,
        "cond_plus_gas_mmr":qt,
        "mean_particle_r":rg*1e4,
        "droplet_eff_r":reff*1e4, 
        "r_units":'micron',
        "column_density":ndz,
        "column_density_unit":'#/cm^2',
        "opd_per_layer":opd, 
        "single_scattering" : w0, 
        "asymmetry": g0, 
        "opd_by_gas": opd_gas,
        "condensibles":gas_names,
        #"scalar_inputs": {'mh':mh, 'mmw':mmw,'fsed':fsed, 'sig':sig,'nrad':nrad,'rmin':rmin},
        "scalar_inputs": {'mh':mh, 'mmw':mmw,'sig':sig,'nrad':nrad,'rmin':rmin},
        "fsed": fsed,
        "altitude":z,
        "layer_thickness":dz_layer,
        "z_unit":'cm',
        'mixing_length':mixl, 
        'mixing_length_unit':'cm',
        'kz':kz, 
        'kz_unit':'cm^2/s',
        'scale_height':scale_h,
        'cloud_deck':z_cld
    }

def calc_optics(nwave, qc, qt, rg, reff, ndz,radius,dr,qext, qscat,cos_qscat,sig, rmin, nrad):
    """
    Calculate spectrally-resolved profiles of optical depth, single-scattering
    albedo, and asymmetry parameter.

    Parameters
    ----------
    nwave : int 
        Number of wave points 
    qc : ndarray
        Condensate mixing ratio 
    qt : ndarray 
        Gas + condensate mixing ratio 
    rg : ndarray 
        Geometric mean radius of condensate 
    reff : ndarray
        Effective (area-weighted) radius of condensate (cm)
    ndz : ndarray
        Column density of particle concentration in layer (#/cm^2)
    radius : ndarray
        Radius bin centers (cm)
    dr : ndarray
        Width of radius bins (cm)
    qscat : ndarray
        Scattering efficiency
    qext : ndarray
        Extinction efficiency
    cos_qscat : ndarray
        qscat-weighted <cos (scattering angle)>
    sig : float 
        Width of the log normal particle distribution


    Returns
    -------
    opd : ndarray 
        extinction optical depth due to all condensates in layer
    w0 : ndarray 
        single scattering albedo
    g0 : ndarray 
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    opd_gas : ndarray
        cumulative (from top) opd by condensing vapor as geometric conservative scatterers
    """

    PI=np.pi
    nz = qc.shape[0]
    ngas = qc.shape[1]
    nrad = len(radius)

    opd_layer = np.zeros((nz, ngas))
    scat_gas = np.zeros((nz,nwave,ngas))
    ext_gas = np.zeros((nz,nwave,ngas))
    cqs_gas = np.zeros((nz,nwave,ngas))
    opd = np.zeros((nz,nwave))
    opd_gas = np.zeros((nz,ngas))
    w0 = np.zeros((nz,nwave))
    g0 = np.zeros((nz,nwave))

    for iz in range(nz):
        for igas in range(ngas):
            # Optical depth for conservative geometric scatterers 
            if ndz[iz,igas] > 0:

#                if np.log10(rg[iz,igas]) < np.log10(rmin)+0.75*sig:
#                    raise Exception ('There has been a calculated particle radii of {0}cm for the {1}th gas at the {2}th grid point. The minimum radius from the Mie grid is {3}cm, and youve requested a lognormal distribution of {4}. Therefore it is not possible to accurately compute the optical properties.'.format(str(rg[iz,igas]),str(igas),str(iz), str(rmin),str(sig)))
#
                r2 = rg[iz,igas]**2 * np.exp( 2*np.log( sig)**2 )
                opd_layer[iz,igas] = 2.*PI*r2*ndz[iz,igas]

                #  Calculate normalization factor (forces lognormal sum = 1.0)
                rsig = sig
                norm = 0.
                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / ( np.sqrt(2.*PI)*rr*np.log(rsig) )
                    arg2 = -np.log( rr/rg[iz,igas] )**2 / ( 2*np.log(rsig)**2 )
                    norm = norm + arg1*np.exp( arg2 )
                    #print (rr, rg[iz,igas],rsig,arg1,arg2)

                # normalization
                norm = ndz[iz,igas] / norm

                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / ( np.sqrt(2.*PI)*np.log(rsig) )
                    arg2 = -np.log( rr/rg[iz,igas] )**2 / ( 2*np.log(rsig)**2 )
                    pir2ndz = norm*PI*rr*arg1*np.exp( arg2 )         
                    for iwave in range(nwave): 
                        scat_gas[iz,iwave,igas] = scat_gas[iz,iwave,igas]+qscat[iwave,irad,igas]*pir2ndz
                        ext_gas[iz,iwave,igas] = ext_gas[iz,iwave,igas]+qext[iwave,irad,igas]*pir2ndz
                        cqs_gas[iz,iwave,igas] = cqs_gas[iz,iwave,igas]+cos_qscat[iwave,irad,igas]*pir2ndz

                    #TO DO ADD IN CLOUD SUBLAYER KLUGE LATER 

    #Sum over gases and compute spectral optical depth profile etc
    for iz in range(nz):
        for iwave in range(nwave): 
            opd_scat = 0.
            opd_ext = 0.
            cos_qs = 0.
            for igas in range(ngas):
                opd_scat = opd_scat + scat_gas[iz,iwave,igas]
                opd_ext = opd_ext + ext_gas[iz,iwave,igas]
                cos_qs = cos_qs + cqs_gas[iz,iwave,igas]

                if( opd_scat > 0. ):
                    opd[iz,iwave] = opd_ext
                    w0[iz,iwave] = opd_scat / opd_ext
                    #if w0[iz,iwave]>1: 
                    #    w0[iz,iwave]=1.
                    g0[iz,iwave] = cos_qs / opd_scat
                    
    #cumulative optical depths for conservative geometric scatterers
    opd_tot = 0.

    for igas in range(ngas):
        opd_gas[0,igas] = opd_layer[0,igas]

        for iz in range(1,nz):
            opd_gas[iz,igas] = opd_gas[iz-1,igas] + opd_layer[iz,igas]

    return opd, w0, g0, opd_gas

def eddysed(t_top, p_top,t_mid, p_mid, condensibles, 
    gas_mw, gas_mmr,rho_p,mw_atmos,gravity, kz,mixl,
    fsed, b, eps, z_top, z_alpha, z_min, param,
    mh,sig, rmin, nrad,d_molecule,eps_k,c_p_factor,
    og_vfall=True,do_virtual=True, supsat=0, verbose=True):
    """
    Given an atmosphere and condensates, calculate size and concentration
    of condensates in balance between eddy diffusion and sedimentation.

    Parameters
    ----------
    t_top : ndarray
        Temperature at each layer (K)
    p_top : ndarray
        Pressure at each layer (dyn/cm^2)
    t_mid : ndarray
        Temperature at each midpoint (K)
    p_mid : ndarray 
        Pressure at each midpoint (dyn/cm^2)
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
        Sedimentation efficiency coefficient, unitless
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_top : float
        Altitude at each layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
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
    og_vfall : bool , optional
        optional, default = True. True does the original fall velocity calculation. 
        False does the updated one which runs a tad slower but is more consistent.
        The main effect of turning on False is particle sizes in the upper atmosphere 
        that are slightly bigger.
    do_virtual : bool,optional 
        optional, Default = True which adds a virtual layer if the 
        species condenses below the model domain.
    supsat : float, optional
        Default = 0 , Saturation factor (after condensation)

    Returns
    -------
    qc : ndarray 
        condenstate mixing ratio (g/g)
    qt : ndarray 
        gas + condensate mixing ratio (g/g)
    rg : ndarray
        geometric mean radius of condensate  cm 
    reff : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz : ndarray 
        number column density of condensate (cm^-3)
    qc_path : ndarray 
        vertical path of condensate 
    """
    #default for everything is false, will fill in as True as we go


    did_gas_condense = [False for i in condensibles]
    t_bot = t_top[-1]
    p_bot = p_top[-1]
    z_bot = z_top[-1]
    ngas =  len(condensibles)
    nz = len(t_mid)
    qc = np.zeros((nz,ngas))
    qt  = np.zeros((nz, ngas))
    rg = np.zeros((nz, ngas))
    reff = np.zeros((nz, ngas))
    ndz = np.zeros((nz, ngas))
    fsed_layer = np.zeros((nz,ngas))
    qc_path = np.zeros(ngas)
    z_cld_out = np.zeros(ngas)

    for i, igas in zip(range(ngas), condensibles):

        q_below = gas_mmr[i]

        #include decrease in condensate mixing ratio below model domain
        if do_virtual: 

            qvs_factor = (supsat+1)*gas_mw[i]/mw_atmos
            get_pvap = getattr(pvaps, igas)
            if igas == 'Mg2SiO4':
                pvap = get_pvap(t_bot, p_bot, mh=mh)
            else:
                pvap = get_pvap(t_bot, mh=mh)

            qvs = qvs_factor*pvap/p_bot   
            if qvs <= q_below :   

                #find the pressure at cloud base 
                #   parameters for finding root 
                p_lo = p_bot
                p_hi = p_bot * 1e3

                #temperature gradient 
                dtdlnp = ( t_top[-2] - t_bot ) / np.log( p_bot/p_top[-2] )

                #   load parameters into qvs_below common block
                qv_dtdlnp = dtdlnp
                qv_p = p_bot
                qv_t = t_bot
                qv_gas_name = igas
                qv_factor = qvs_factor

                try:

                    p_base = optimize.root_scalar(qvs_below_model, 
                                bracket=[p_lo, p_hi], method='brentq', 
                                args=(qv_dtdlnp,qv_p, qv_t,qv_factor ,qv_gas_name,mh,q_below)
                                )#, xtol = 1e-20)

                    if verbose: print('Virtual Cloud Found: '+ qv_gas_name)
                    root_was_found = True
                except ValueError: 
                    root_was_found = False

                if root_was_found:
                    #Yes, the gas did condense (below the grid)
                    did_gas_condense[i] = True

                    p_base = p_base.root 
                    t_base = t_bot + np.log( p_bot/p_base )*dtdlnp
                    z_base = z_bot + scale_h * np.log( p_bot_sub/p_base ) 
                    
                    #   Calculate temperature and pressure below bottom layer
                    #   by adding a virtual layer 

                    p_layer_virtual = 0.5*( p_bot + p_base )
                    t_layer_virtual = t_bot + np.log10( p_bot/p_layer_virtual )*dtdlnp

                    #we just need to overwrite 
                    #q_below from this output for the next routine
                    qc_v, qt_v, rg_v, reff_v,ndz_v,q_below, z_cld, fsed_layer_v = layer( igas, rho_p[i], 
                        #t,p layers, then t.p levels below and above
                        t_layer_virtual, p_layer_virtual, t_bot,t_base, p_bot, p_base,
                        kz[-1], mixl[-1], gravity, mw_atmos, gas_mw[i], q_below,
                        supsat, fsed, b, eps, z_bot, z_base, z_alpha, z_min, param,
                        sig,mh, rmin, nrad, d_molecule,eps_k,c_p_factor, #all scalaers
                        og_vfall, z_cld
                    )

        z_cld=None
        for iz in range(nz-1,-1,-1): #goes from BOA to TOA

            qc[iz,i], qt[iz,i], rg[iz,i], reff[iz,i],ndz[iz,i],q_below, z_cld, fsed_layer[iz,i]  = layer( igas, rho_p[i], 
                #t,p layers, then t.p levels below and above
                t_mid[iz], p_mid[iz], t_top[iz], t_top[iz+1], p_top[iz], p_top[iz+1],
                kz[iz], mixl[iz], gravity, mw_atmos, gas_mw[i], q_below,  
                supsat, fsed, b, eps, z_top[iz], z_top[iz+1], z_alpha, z_min, param,
                sig,mh, rmin, nrad, d_molecule,eps_k,c_p_factor, #all scalars
                og_vfall, z_cld
            )

            qc_path[i] = (qc_path[i] + qc[iz,i]*
                            ( p_top[iz+1] - p_top[iz] ) / gravity)
        z_cld_out[i] = z_cld

    return qc, qt, rg, reff, ndz, qc_path,mixl, z_cld_out

def layer(gas_name,rho_p, t_layer, p_layer, t_top, t_bot, p_top, p_bot,
    kz, mixl, gravity, mw_atmos, gas_mw, q_below,
    supsat, fsed, b, eps, z_top, z_bot, z_alpha, z_min, param,
    sig,mh, rmin, nrad, d_molecule,eps_k,c_p_factor,
    og_vfall, z_cld):
    """
    Calculate layer condensate properties by iterating on optical depth
    in one model layer (convering on optical depth over sublayers)

    gas_name : str 
        Name of condenstante 
    rho_p : float 
        density of condensed vapor (g/cm^3)
    t_layer : float 
        Temperature of layer mid-pt (K)
    p_layer : float 
        Pressure of layer mid-pt (dyne/cm^2)
    t_top : float 
        Temperature at top of layer (K)
    t_bot : float 
        Temperature at botton of layer (K)
    p_top : float 
        Pressure at top of layer (dyne/cm2)
    p_bot : float 
        Pressure at botton of layer 
    kz : float 
        eddy diffusion coefficient (cm^2/s)
    mixl : float 
        Mixing length (cm)
    gravity : float 
        Gravity of planet cgs 
    mw_atmos : float 
        Molecular weight of the atmosphere 
    gas_mw : float 
        Gas molecular weight 
    q_below : float 
        total mixing ratio (vapor+condensate) below layer (g/g)
    supsat : float 
        Super saturation factor
    fsed : float
        Sedimentation efficiency coefficient (unitless) 
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_top : float
        Altitude at top of layer
    z_bot : float
        Altitude at bottom of layer
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float 
        Width of the log normal particle distribution 
    mh : float 
        Metallicity NON log soar (1=1xSolar)
    rmin : float 
        Minium radius on grid (cm)
    nrad : int 
        Number of radii on Mie grid
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
    og_vfall : bool 
        Use original or new vfall calculation

    Returns
    -------
    qc_layer : ndarray 
        condenstate mixing ratio (g/g)
    qt_layer : ndarray 
        gas + condensate mixing ratio (g/g)
    rg_layer : ndarray
        geometric mean radius of condensate  cm 
    reff_layer : ndarray
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : ndarray 
        number column density of condensate (cm^-3)
    q_below : ndarray 
        total mixing ratio (vapor+condensate) below layer (g/g)
    """
    #   universal gas constant (erg/mol/K)
    nsub_max = 128
    R_GAS = 8.3143e7
    AVOGADRO = 6.02e23
    K_BOLTZ = R_GAS / AVOGADRO
    PI = np.pi 
    #   Number of levels of grid refinement used 
    nsub = 1

    #   specific gas constant for atmosphere (erg/K/g)
    r_atmos = R_GAS / mw_atmos

    #specific gas constant for cloud (erg/K/g)
    r_cloud = R_GAS/ gas_mw

    #   specific heat of atmosphere (erg/K/g)
    c_p = c_p_factor * r_atmos

    #   pressure thickness of layer
    dp_layer = p_bot - p_top
    dlnp = np.log( p_bot/p_top )

    #   temperature gradient 
    dtdlnp = ( t_top - t_bot ) / dlnp
    lapse_ratio = ( t_bot - t_top ) / dlnp / ( t_layer / c_p_factor )

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / ( r_atmos * t_layer )

    #   atmospheric scale height (cm)
    scale_h = r_atmos * t_layer / gravity    

    #   convective velocity scale (cm/s) from mixing length theory
    w_convect = kz / mixl 

    #   atmospheric number density (molecules/cm^3)
    n_atmos = p_layer / ( K_BOLTZ*t_layer )

    #   atmospheric mean free path (cm)
    mfp = 1. / ( np.sqrt(2.)*n_atmos*PI*d_molecule**2 )

    # atmospheric viscosity (dyne s/cm^2)
    # EQN B2 in A & M 2001, originally from Rosner+2000
    # Rosner, D. E. 2000, Transport Processes in Chemically Reacting Flow Systems (Dover: Mineola)
    visc = (5./16.*np.sqrt( PI*K_BOLTZ*t_layer*(mw_atmos/AVOGADRO)) /
        ( PI*d_molecule**2 ) /
        ( 1.22 * ( t_layer / eps_k )**(-0.16) ))


    #   --------------------------------------------------------------------
    #   Top of convergence loop    
    converge = False
    while not converge: 

        #   Zero cumulative values
        qc_layer = 0.
        qt_layer = 0.
        ndz_layer = 0.
        opd_layer = 0.        

        #   total mixing ratio and pressure at bottom of sub-layer

        qt_bot_sub = q_below
        p_bot_sub = p_bot
        z_bot_sub = z_bot

        #SUBALYER 
        dp_sub = dp_layer / nsub

        for isub in range(nsub): 
            qt_below = qt_bot_sub
            p_top_sub = p_bot_sub - dp_sub
            dz_sub = scale_h * np.log( p_bot_sub/p_top_sub ) # width of layer
            p_sub = 0.5*( p_bot_sub + p_top_sub )
            #################### CHECK #####################
            z_top_sub = z_bot_sub + dz_sub
            z_sub = z_bot_sub + scale_h * np.log( p_bot_sub/p_sub ) # midpoint of layer 
            ################################################
            t_sub = t_bot + np.log( p_bot/p_sub )*dtdlnp
            qt_top, qc_sub, qt_sub, rg_sub, reff_sub,ndz_sub, z_cld, fsed_layer = calc_qc(
                    gas_name, supsat, t_sub, p_sub,r_atmos, r_cloud,
                        qt_below, mixl, dz_sub, gravity,mw_atmos,mfp,visc,
                        rho_p,w_convect, fsed, b, eps, param, z_bot_sub, z_sub, z_alpha, z_min,
                        sig,mh, rmin, nrad, og_vfall,z_cld)


            #   vertical sums
            qc_layer = qc_layer + qc_sub*dp_sub/gravity
            qt_layer = qt_layer + qt_sub*dp_sub/gravity
            ndz_layer = ndz_layer + ndz_sub

            if reff_sub > 0.:
                opd_layer = (opd_layer + 
                                    1.5*qc_sub*dp_sub/gravity/(rho_p*reff_sub))
    
            #   Increment values at bottom of sub-layer

            qt_bot_sub = qt_top
            p_bot_sub = p_top_sub
            z_bot_sub = z_top_sub

        #    Check convergence on optical depth
        if nsub_max == 1 :
            converge = True
        elif  nsub == 1 : 
            opd_test = opd_layer
        elif (opd_layer == 0.) or (nsub >= nsub_max): 
            converge = True
        elif ( abs( 1. - opd_test/opd_layer ) <= 1e-2 ) : 
            converge = True
        else: 
            opd_test = opd_layer
        
        nsub = nsub * 2
    #   Update properties at bottom of next layer

    q_below = qt_top

    #Get layer averages

    if opd_layer > 0. : 
        reff_layer = 1.5*qc_layer / (rho_p*opd_layer)
        lnsig2 = 0.5*np.log( sig )**2
        rg_layer = reff_layer*np.exp( -5*lnsig2 )
    else : 
        reff_layer = 0.
        rg_layer = 0.

    qc_layer = qc_layer*gravity / dp_layer
    qt_layer = qt_layer*gravity / dp_layer

    return qc_layer, qt_layer, rg_layer, reff_layer, ndz_layer,q_below, z_cld, fsed_layer

def calc_qc(gas_name, supsat, t_layer, p_layer
    ,r_atmos, r_cloud, q_below, mixl, dz_layer, gravity,mw_atmos
    ,mfp,visc,rho_p,w_convect, fsed, b, eps, param, z_bot, z_layer, z_alpha, z_min,
    sig, mh, rmin, nrad, og_vfall=True, z_cld=None):
    """
    Calculate condensate optical depth and effective radius for a layer,
    assuming geometric scatterers. 

    gas_name : str 
        Name of condensate 
    supsat : float 
        Super saturation factor 
    t_layer : float 
        Temperature of layer mid-pt (K)
    p_layer : float 
        Pressure of layer mid-pt (dyne/cm^2)
    r_atmos : float 
        specific gas constant for atmosphere (erg/K/g)
    r_cloud : float 
        specific gas constant for cloud species (erg/K/g)     
    q_below : float 
        total mixing ratio (vapor+condensate) below layer (g/g)
    mxl : float 
        convective mixing length scale (cm): no less than 1/10 scale height
    dz_layer : float 
        Thickness of layer cm 
    gravity : float 
        Gravity of planet cgs 
    mw_atmos : float 
        Molecular weight of the atmosphere 
    mfp : float 
        atmospheric mean free path (cm)
    visc : float 
        atmospheric viscosity (dyne s/cm^2)
    rho_p : float 
        density of condensed vapor (g/cm^3)
    w_convect : float    
        convective velocity scale (cm/s)
    fsed : float
        Sedimentation efficiency coefficient (unitless) 
    b : float
        Denominator of exponential in sedimentation efficiency  (if param is 'exp')
    eps: float
        Minimum value of fsed function (if param=exp)
    z_bot : float
        Altitude at bottom of layer
    z_layer : float 
        Altitude of midpoint of layer (cm)
    z_alpha : float
        Altitude at which fsed=alpha for variable fsed calculation
    param : str
        fsed parameterisation
        'const' (constant), 'exp' (exponential density derivation)
    sig : float 
        Width of the log normal particle distrubtion 
    mh : float 
        Metallicity NON log solar (1 = 1x solar)
    rmin : float 
        Minium radius on grid (cm)
    nrad : int 
        Number of radii on Mie grid

    Returns
    -------
    qt_top : float 
        gas + condensate mixing ratio at top of layer(g/g)
    qc_layer : float 
        condenstate mixing ratio (g/g)
    qt_layer : float 
        gas + condensate mixing ratio (g/g)
    rg_layer : float
        geometric mean radius of condensate  cm 
    reff_layer : float
        droplet effective radius (second moment of size distrib, cm)
    ndz_layer : float 
        number column density of condensate (cm^-3)
    """

    get_pvap = getattr(pvaps, gas_name)
    if gas_name == 'Mg2SiO4':
        pvap = get_pvap(t_layer, p_layer,mh=mh)
    else:
        pvap = get_pvap(t_layer,mh=mh)

    fs = supsat + 1 

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / ( r_atmos * t_layer )    

    #   mass mixing ratio of saturated vapor (g/g)
    qvs = fs*pvap / ( (r_cloud) * t_layer ) / rho_atmos

    #   --------------------------------------------------------------------
    #   Layer is cloud free

    if( q_below < qvs ):

        qt_layer = q_below
        qt_top   = q_below
        qc_layer = 0.
        rg_layer = 0.
        reff_layer = 0.
        ndz_layer = 0.
        z_cld = z_cld
        fsed_mid = 0

    else:

        #   --------------------------------------------------------------------
        #   Cloudy layer: first calculate qt and qc at top of layer,
        #   then calculate layer averages
        if z_cld is None:
            z_cld = z_bot
        else:
            z_cld = z_cld

        #   range of mixing ratios to search (g/g)
        qhi = q_below
        qlo = qhi / 1e3

        #   load parameters into advdiff common block

        ad_qbelow = q_below
        ad_qvs = qvs
        ad_mixl = mixl
        ad_dz = dz_layer
        ad_rainf = fsed

        #   Find total vapor mixing ratio at top of layer
        #find_root = True
        #while find_root:
        #    try:
        #        qt_top = optimize.root_scalar(advdiff, bracket=[qlo, qhi], method='brentq',
        #                    args=(ad_qbelow,ad_qvs, ad_mixl,ad_dz ,ad_rainf,
        #                        z_bot, b, eps, param)
        #                        )#, xtol = 1e-20)
        #        find_root = False
        #    except ValueError:
        #        qlo = qlo/10
        #
        #qt_top = qt_top.root
        if param is "const":
            qt_top = qvs + (q_below - qvs) * np.exp(-fsed * dz_layer / mixl)
        elif param is "exp":
            fs = fsed / np.exp(z_alpha / b)
            qt_top = qvs + (q_below - qvs) * np.exp( - b * fs / mixl * np.exp(z_bot/b) 
                            * (np.exp(dz_layer/b) -1) + eps*dz_layer/mixl)

        #   Use trapezoid rule (for now) to calculate layer averages
        #   -- should integrate exponential
        qt_layer = 0.5*( q_below + qt_top )


        #   Find total condensate mixing ratio
        qc_layer = np.max( [0., qt_layer - qvs] )

        #   --------------------------------------------------------------------
        #   Find <rw> corresponding to <w_convect> using function vfall()

        #   range of particle radii to search (cm)
        rlo = 1.e-10
        rhi = 10.
        
        #   precision of vfall solution (cm/s)
        find_root = True
        while find_root:
            try:
                if og_vfall:
                    rw_temp = optimize.root_scalar(vfall_find_root, bracket=[rlo, rhi], method='brentq', 
                            args=(gravity,mw_atmos,mfp,visc,t_layer,p_layer, rho_p,w_convect))
                else:
                    rw_temp = solve_force_balance("rw", w_convect, gravity, mw_atmos, mfp,
                                                    visc, t_layer, p_layer, rho_p, rlo, rhi)
                find_root = False
            except ValueError:
                rlo = rlo/10
                rhi = rhi*10

        #fall velocity particle radius 
        if og_vfall: rw_layer = rw_temp.root
        else: rw_layer = rw_temp
        
        #   geometric std dev of lognormal size distribution
        lnsig2 = 0.5*np.log( sig )**2
        #   sigma floor for the purpose of alpha calculation
        sig_alpha = np.max( [1.1, sig] )    

        #   find alpha for power law fit vf = w(r/rw)^alpha
        def pow_law(r, alpha):
            return np.log(w_convect) + alpha * np.log (r / rw_layer) 

        r_, rup, dr = get_r_grid(r_min = rmin, n_radii = nrad)
        vfall_temp = []
        for j in range(len(r_)):
            if og_vfall:
                vfall_temp.append(vfall(r_[j], gravity, mw_atmos, mfp, visc, t_layer, p_layer, rho_p))
            else:
                vlo = 1e0; vhi = 1e6
                find_root = True
                while find_root:
                    try:
                        vfall_temp.append(solve_force_balance("vfall", r_[j], gravity, mw_atmos, 
                            mfp, visc, t_layer, p_layer, rho_p, vlo, vhi))
                        find_root = False
                    except ValueError:
                        vlo = vlo/10
                        vhi = vhi*10

        pars, cov = optimize.curve_fit(f=pow_law, xdata=r_, ydata=np.log(vfall_temp), p0=[0], 
                            bounds=(-np.inf, np.inf))
        alpha = pars[0]


        #   fsed at middle of layer 
        if param is 'exp':
            fsed_mid = fs * np.exp(z_layer / b) + eps
        else: # 'const'
            fsed_mid = fsed

        #     EQN. 13 A&M 
        #   geometric mean radius of lognormal size distribution
        rg_layer = (fsed_mid**(1./alpha) *
                    rw_layer * np.exp( -(alpha+6)*lnsig2 ))

        #   droplet effective radius (cm)
        reff_layer = rg_layer*np.exp( 5*lnsig2 )

        #      EQN. 14 A&M
        #   column droplet number concentration (cm^-2)
        ndz_layer = (3*rho_atmos*qc_layer*dz_layer /
                    ( 4*np.pi*rho_p*rg_layer**3 ) * np.exp( -9*lnsig2 ))

    return qt_top, qc_layer,qt_layer, rg_layer,reff_layer,ndz_layer, z_cld, fsed_mid 

class Atmosphere():
    def __init__(self,condensibles, fsed=0.5, b=1, eps=1e-2, mh=1, mmw=2.2, sig=2.0,
                    param='const', verbose=True, supsat=0, gas_mmr=None):
        """
        Parameters
        ----------
        condensibles : list of str
            list of gases for which to consider as cloud species 
        fsed : float 
            Sedimentation efficiency coefficient. Jupiter ~3-6. Hot Jupiters ~ 0.1-1.
        b : float
            Denominator of exponential in sedimentation efficiency  (if param is 'exp')
        eps: float
            Minimum value of fsed function (if param=exp)
        mh : float 
            metalicity 
        mmw : float 
            MMW of the atmosphere 
        sig : float 
            Width of the log normal distribution for the particle sizes 
        param : str
            fsed parameterisation
            'const' (constant), 'exp' (exponential density derivation)
        verbose : bool 
            Prints out warning statements throughout
    
        """
        self.mh = mh
        self.mmw = mmw
        self.condensibles = condensibles
        self.fsed = fsed
        self.b = b
        self.sig = sig
        self.param = param
        self.eps = eps
        self.verbose = verbose 
        #grab constants
        self.constants()
        self.supsat = supsat
        self.gas_mmr = gas_mmr

    def constants(self):
        #   Depth of the Lennard-Jones potential well for the atmosphere 
        # Used in the viscocity calculation (units are K) (Rosner, 2000)
        #   (78.6 for air, 71.4 for N2, 59.7 for H2)
        self.eps_k = 59.7
        #   diameter of atmospheric molecule (cm) (Rosner, 2000)
        #   (3.711e-8 for air, 3.798e-8 for N2, 2.827e-8 for H2)
        self.d_molecule = 2.827e-8

        #specific heat factor of the atmosphere 
        #7/2 comes from ideal gas assumption of permanent diatomic gas 
        #e.g. h2, o2, n2, air, no, co
        #this technically does increase slowly toward higher temperatures (>~700K)
        self.c_p_factor = 7./2.

        self.R_GAS = 8.3143e7
        self.AVOGADRO = 6.02e23
        self.K_BOLTZ = self.R_GAS / self.AVOGADRO

    def ptk(self, df = None, filename=None,
        kz_min=1e5, constant_kz=None, latent_heat=False, convective_overshoot=None,
        Teff=None, alpha_pressure=None, **pd_kwargs):
        """
        Read in file or define dataframe. 
    
        Parameters
        ----------
        df : dataframe or dict
            Dataframe with "pressure"(bars),"temperature"(K). MUST have at least two 
            columns with names "pressure" and "temperature". 
            Optional columns include the eddy diffusion "kz" in cm^2/s CGS units, and 
            the convective heat flux 'chf' also in cgs (e.g. sigma_csg T^4)
        filename : str 
            Filename read in. Will be read in with pd.read_csv and should 
            result in two named headers "pressure"(bars),"temperature"(K). 
            Optional columns include the eddy diffusion "kz" in cm^2/s CGS units, and 
            the convective heat flux 'chf' also in cgs (e.g. sigma_csg T^4)
            Use pd_kwargs to ensure file is read in properly.
        kz_min : float, optional
            Minimum Kz value. This will reset everything below kz_min to kz_min. 
            Default = 1e5 cm2/s
        constant_kz : float, optional
            Constant value for kz, if kz is supplied in df or filename, 
            it will inheret that value and not use this constant_value
            Default = None 
        latent_heat : bool 
            optional, Default = False. The latent heat factors into the mixing length. 
            When False, the mixing length goes as the scale height 
            When True, the mixing length is scaled by the latent heat 
        convective_overshoot : float 
            Optional, Default is None. But the default value used in 
            Ackerman & Marley 2001 is 1./3. If you are unsure of what to pick, start 
            there. This is only used when the        
            This is ONLY used when a chf (convective heat flux) is supplied 
        Teff : float, optional
            Effective temperature. If None (default), Teff set to temperature at 1 bar
        alpha_pressure : float
            Pressure at which we want fsed=alpha for variable fsed calculation
        pd_kwargs : kwargs
            Pandas key words for file read in. 
            If reading old style eddysed files, you would need: 
            skiprows=3, delim_whitespace=True, header=None, names=["ind","pressure","temperature","kz"]
        """
        #first read in dataframe, dict or file and sort by pressure
        if not isinstance(df, type(None)):
            if isinstance(df, dict): df = pd.DataFrame(df)
            df = df.sort_values('pressure')
        elif not isinstance(filename, type(None)):
            df = pd.read_csv(filename, **pd_kwargs)
            df = df.sort_values('pressure')

        #convert bars to dyne/cm^2 
        self.p_level = np.array(df['pressure'])*1e6
        self.t_level = np.array(df['temperature'])      
        if alpha_pressure is None:
            self.alpha_pressure=min(df['pressure'])
        else:
            self.alpha_pressure=alpha_pressure
        self.get_atmo_parameters()
        self.get_kz_mixl(df, constant_kz, latent_heat, convective_overshoot, kz_min)

        # Teff
        if isinstance(Teff, type(None)):
            onebar = (np.abs(self.p_level/1e6 - 1.)).argmin()
            self.Teff = self.t_level[onebar]
        else:
            self.Teff = Teff



    def get_atmo_parameters(self):
        """Defines all the atmospheric parameters needed for the calculation

        Note: Some of this is repeated in the layer() function. 
        This is done on purpose because layer is used to get a "virtual"
        layer off the user input's grid. These parameters are also 
        needed, though, to get the initial mixing parameters from the user. 
        Therefore, we put them in both places. 
        """
        #   specific gas constant for atmosphere (erg/K/g)
        self.r_atmos = self.R_GAS / self.mmw
        
        # pressure thickess 
        dlnp = np.log( self.p_level[1:] / self.p_level[0:-1] ) #USED IN LAYER
        self.p_layer = 0.5*( self.p_level[1:] + self.p_level[0:-1]) #USED IN LAYER
        
        #   temperature gradient - we use this for the sub layering
        self.dtdlnp = ( self.t_level[0:-1] - self.t_level[1:] ) / dlnp #USED IN LAYER
        
        # get temperatures at layers
        self.t_layer = self.t_level[1:] + np.log( self.p_level[1:]/self.p_layer )*self.dtdlnp #USED IN LAYER

        #lapse ratio used for kz calculation if user asks for 
        # us ot calculate it based on convective heat flux
        self.lapse_ratio = ( self.t_level[1:] - self.t_level[0:-1] 
                            ) / dlnp / ( self.t_layer/self.c_p_factor )

        #   atmospheric density (g/cm^3)
        self.rho_atmos = self.p_layer / (self.r_atmos * self.t_layer)

        #scale height (todo=make this gravity dependent)
        self.scale_h = self.r_atmos * self.t_layer / self.g

        #specific heat of atmosphere
        self.c_p = self.c_p_factor * self.r_atmos

        #get altitudes 
        self.dz_pmid = self.scale_h * np.log( self.p_level[1:]/self.p_layer )

        self.dz_layer = self.scale_h * dlnp

        self.z_top = np.concatenate(([0],np.cumsum(self.dz_layer[::-1])))[::-1]

        self.z = self.z_top[1:]+self.dz_pmid

        # altitude to set fsed = alpha
        p_alpha = find_nearest_1d(self.p_layer/1e6, self.alpha_pressure)
        z_temp = np.cumsum(self.dz_layer[::-1])[::-1]
        self.z_alpha = z_temp[p_alpha]

    def get_kz_mixl(self, df, constant_kz, latent_heat, convective_overshoot,
     kz_min):
        """
        Computes kz profile and mixing length given user input. In brief the options are: 

        1) Input Kz
        2) Input constant kz
        3) Input convective heat flux (supply chf in df)
        3a) Input convective heat flux, correct for latent heat (supply chf in df and set latent_heat=True)
        and/or 3b) Input convective heat flux, correct for convective overshoot (supply chf, convective_overshoot=1/3)
        4) Set kz_min to prevent kz from going too low (any of the above and set kz_min~1e5)

        Parameters
        ----------
        df : dataframe or dict
            Dataframe from input with "pressure"(bars),"temperature"(K). MUST have at least two 
            columns with names "pressure" and "temperature". 
            Optional columns include the eddy diffusion "kz" in cm^2/s CGS units, and 
            the convective heat flux 'chf' also in cgs (e.g. sigma_csg T^4)
        constant_kz : float
            Constant value for kz, if kz is supplied in df or filename, 
            it will inheret that value and not use this constant_value
            Default = None 
        latent_heat : bool 
            optional, Default = False. The latent heat factors into the mixing length. 
            When False, the mixing length goes as the scale height 
            When True, the mixing length is scaled by the latent heat 
        convective_overshoot : float 
            Optional, Default is None. But the default value used in 
            Ackerman & Marley 2001 is 1./3. If you are unsure of what to pick, start 
            there. This is only used when the        
            This is ONLY used when a chf (convective heat flux) is supplied 
        kz_min : float
            Minimum Kz value. This will reset everything below kz_min to kz_min. 
            Default = 1e5 cm2/s
        """

        #MIXING LENGTH ASSUMPTIONS 
        if latent_heat:
            #   convective mixing length scale (cm): no less than 1/10 scale height
            self.mixl = np.array([np.max( [0.1, ilr] ) for ilr in self.lapse_ratio]) * self.scale_h
        else:
            #convective mixing length is the scale height
            self.mixl = 1 * self.scale_h


        #KZ OPTIONS 

        #   option 1) the user has supplied it in their file or dictionary
        if 'kz' in df.keys(): 
            if df.loc[df['kz']<kz_min].shape[0] > 0:
                df.loc[df['kz']<kz_min] = kz_min
                if self.verbose: print('Overwriting some Kz values to minimum value set by kz_min \n \
                    You can always turn off these warnings by setting verbose=False') 
            kz_level = np.array(df['kz'])
            self.kz = 0.5*(kz_level[1:] + kz_level[0:-1])
            self.chf = None

        #   option 2) the user wants a constant value
        elif not isinstance(constant_kz , type(None)):
            self.kz = np.zeros(df.shape[0]-1) + constant_kz
            self.chf = None

        #   option 3) the user wants to compute kz based on a convective heat flux 
        elif 'chf' in df.keys():
            self.chf =  np.array(df['chf'])


            #CONVECTIVE OVERSHOOT ON OR OFF
            #     sets the minimum allowed heat flux in a layer by assuming some overshoot
            #     the default value of 1/3 is arbitrary, allowing convective flux to fall faster than
            #     pressure scale height
            if not isinstance(convective_overshoot, type(None)):
                used = False
                nz = len(self.p_layer)
                for iz in range(nz-1,-1,-1):
                    ratio_min = (convective_overshoot)*self.p_level[iz]/self.p_level[iz+1] 
                    if self.chf[iz] < ratio_min*self.chf[iz+1]:
                        self.chf[iz] = self.chf[iz+1]*ratio_min
                        used=True
                if self.verbose: print('Convective overshoot was turned on. The convective heat flux \n \
                    has been adjusted such that it is not allowed to decrease more than {0} \n \
                    the pressure. This number is set with the convective_overshoot parameter. \n \
                    It can be disabled with convective_overshoot=None. To turn \n \
                    off these messages set verbose=False in Atmosphere'.format(convective_overshoot)) 

            #   vertical eddy diffusion coefficient (cm^2/s)
            #   from Gierasch and Conrath (1985)
            gc_kzz = ((1./3.) * self.scale_h * (self.mixl/self.scale_h)**(4./3.) * 
                    ( ( self.r_atmos*self.chf[1:] ) / ( self.rho_atmos*self.c_p  ) )**(1./3.)) 
            
            self.kz =  [np.max([i, kz_min]) for i in gc_kzz ]
        else:
            raise Exception("Users can define kz by: \n \
            1) Adding 'kz' as a column or key to your dataframe dict, or file \n \
            2) Defining constant-w-altitude kz through the constant_kz input \n  \
            3) Adding 'chf', the conective heat flux as a column to your \
            dataframe, dict or file.")

    def gravity(self, gravity=None, gravity_unit=None, radius=None, 
        radius_unit=None, mass = None, mass_unit=None):
        """
        Get gravity based on mass and radius, or gravity inputs 

        Parameters
        ----------
        gravity : float 
            (Optional) Gravity of planet 
        gravity_unit : astropy.unit
            (Optional) Unit of Gravity
        radius : float 
            (Optional) radius of planet MUST be specified for thermal emission!
        radius_unit : astropy.unit
            (Optional) Unit of radius
        mass : float 
            (Optional) mass of planet 
        mass_unit : astropy.unit
            (Optional) Unit of mass   
        """
        if (mass is not None) and (radius is not None):
            m = (mass*mass_unit).to(u.g)
            r = (radius*radius_unit).to(u.cm)
            g = (c.G.cgs * m /  (r**2)).value
            self.g = g
            self.gravity_unit = 'cm/(s**2)'
        elif gravity is not None:
            g = (gravity*gravity_unit).to('cm/(s**2)')
            g = g.value
            self.g = g
            self.gravity_unit = 'cm/(s**2)'
        else: 
            raise Exception('Need to specify gravity or radius and mass + additional units')


    def kz(self,df = None, constant_kz=None, chf = None, kz_min = 1e5, latent_heat=False): 
        """
        Define Kz in CGS. Should be on same grid as pressure. This overwrites whatever was 
        defined in get_pt ! Users can define kz by: 
            1) Defining a DataFrame with keys 'pressure' (in bars), and 'kz'
            2) Defining constant kz 
            3) Supplying a convective heat flux and prescription for latent_heat

        Parameters
        ----------
        df : pandas.DataFrame, dict
            Dataframe or dictionary with 'kz' as one of the fields. 
        constant_kz : float 
            Constant value for kz in units of cm^2/s
        chf : ndarray 
            Convective heat flux in cgs units (e.g. sigma T^4). This will be used to compute 
            the kzz using the methodology of Gierasch and Conrath (1985)
        latent_heat : bool 
            optional, Default = False. The latent heat factors into the mixing length. 
            When False, the mixing length goes as the scale height 
            When True, the mixing length is scaled by the latent heat 
            This is ONLY used when a chf (convective heat flux) is supplied 
        """
        return "Depricating this function. Please use ptk instead. It has identical functionality."
        if not isinstance(df, type(None)):
            #will not need any convective heat flux 
            self.chf = None
            #reset to minimun value if specified by the user
            if df.loc[df['kz']<kz_min].shape[0] > 0:
                df.loc[df['kz']<kz_min] = kz_min
                print('Overwriting some Kz values to minimum value set by kz_min') 
            self.kz = np.array(df['kz'])
            #make sure pressure and kz are the same size 
            if len(self.kz) != len(self.pressure) : 
                raise Exception('Kzz and pressure are not the same length')

        elif not isinstance(constant_kz, type(None)):
            #will not need any convective heat flux
            self.chf = None
            self.kz = constant_kz
            if self.kz<kz_min:
                self.kz = kz_min
                print('Overwriting kz constant value to minimum value set by kz_min')

        elif not isinstance(chf, type(None)):
            def g_c_85(scale_h,r_atmos, chf, rho_atmos, c_p, lapse_ratio):
                #   convective mixing length scale (cm): no less than 1/10 scale height
                if latent_heat:
                    mixl = np.max( 0.1, lapse_ratio ) * scale_h
                else:
                    mixl = scale_h
                #   vertical eddy diffusion coefficient (cm^2/s)
                #   from Gierasch and Conrath (1985)
                gc_kzz = ((1./3.) * scale_h * (mixl/scale_h)**(4./3.) * 
                        ( ( r_atmos*chf ) / ( rho_atmos*c_p ) )**(1./3.)) 
                return np.max(gc_kzz, kz_min), mixl

            self.kz = g_c_85
            self.chf = chf

    def compute(self,directory = None, as_dict = True): 
        """
        Parameters
        ----------
        atmo : class 
            `Atmosphere` class 
        directory : str, optional 
            Directory string that describes where refrind files are 
        as_dict : bool 
            Default = True, option to view full output as dictionary

        Returns 
        -------
        dict 
            When as_dict=True. Dictionary output that contains full output. See tutorials for explanation of all output.        
        opd, w0, g0
            Extinction per layer, single scattering abledo, asymmetry parameter, 
            All are ndarrays that are nlayer by nwave
        """
        run = compute(self, directory = directory, as_dict = as_dict)
        return run

def calc_mie_db(gas_name, dir_refrind, dir_out, rmin = 1e-8, nradii = 60):
    """
    Function that calculations new Mie database using PyMieScatt. 

    Parameters
    ----------
    gas_name : list, str
        List of names of gasses. Or a single gas name. 
        See pyeddy.available() to see which ones are currently available. 
    dir_refrind : str 
        Directory where you store optical refractive index files that will be created. 
    dir_out: str 
        Directory where you want to store Mie parameter files. Will be stored as gas_name.Mieff. 
        BEWARE FILE OVERWRITES. 
    rmin : float , optional
        (Default=1e-5) Units of cm. The minimum radius to compute Mie parameters for. 
        Usually 0.1 microns is small enough. However, if you notice your mean particle radius 
        is on the low end, you may compute your grid to even lower particle sizes. 
    nradii : int, optional
        (Default=40) number of radii points to compute grid on. 40 grid points for exoplanets/BDs
        is generally sufficient. 

    Returns 
    -------
    Q extinction, Q scattering,  asymmetry * Q scattering, radius grid (cm), wavelength grid (um)

    The Q "efficiency factors" are = cross section / geometric cross section of particle
    """
    if isinstance(gas_name,str):
        gas_name = [gas_name]
    ngas = len(gas_name)

    for i in range(len(gas_name)): 
        print('Computing ' + gas_name[i])
        #Setup up a particle size grid on first run and calculate single-particle scattering
        
        #files will be saved in `directory`
        # obtaining refractive index data for each gas
        wave_in,nn,kk = get_refrind(gas_name[i],dir_refrind)
        nwave = len(wave_in)

        if i==0:
            #all these files need to be on the same grid
            radius, rup, dr = get_r_grid(r_min = rmin, n_radii = nradii)

            qext_all=np.zeros(shape=(nwave,nradii,ngas))
            qscat_all = np.zeros(shape=(nwave,nradii,ngas))
            cos_qscat_all=np.zeros(shape=(nwave,nradii,ngas))

        #get extinction, scattering, and asymmetry
        #all of these are  [nwave by nradii]
        qext_gas, qscat_gas, cos_qscat_gas = calc_new_mieff(wave_in, nn,kk, radius, rup, fort_calc_mie = False)

        #add to master matrix that contains the per gas Mie stuff
        qext_all[:,:,i], qscat_all[:,:,i], cos_qscat_all[:,:,i] = qext_gas, qscat_gas, cos_qscat_gas 

        #prepare format for old ass style
        wave = [nwave] + sum([[r]+list(wave_in) for r in radius],[])
        qscat = [nradii]  + sum([[np.nan]+list(iscat) for iscat in qscat_gas.T],[])
        qext = [np.nan]  + sum([[np.nan]+list(iext) for iext in qext_gas.T],[])
        cos_qscat = [np.nan]  + sum([[np.nan]+list(icos) for icos in cos_qscat_gas.T],[])

        pd.DataFrame({'wave':wave,'qscat':qscat,'qext':qext,'cos_qscat':cos_qscat}).to_csv(os.path.join(dir_out,gas_name[i]+".mieff"),
                                                                                   sep=' ',
                                                                                  index=False,header=None)
    return qext_all, qscat_all, cos_qscat_all, radius,wave_in

def get_mie(gas, directory):
    """
    Get Mie parameters from old ass formatted files
    """
    df = pd.read_csv(os.path.join(directory,gas+".mieff"),names=['wave','qscat','qext','cos_qscat'], delim_whitespace=True)

    nwave = int( df.iloc[0,0])
    nradii = int(df.iloc[0,1])

    #get the radii (all the rows where there the last three rows are nans)
    radii = df.loc[np.isnan(df['qscat'])]['wave'].values

    df = df.dropna()

    assert len(radii) == nradii , "Number of radii specified in header is not the same as number of radii."
    assert nwave*nradii == df.shape[0] , "Number of wavelength specified in header is not the same as number of waves in file"

    wave = df['wave'].values.reshape((nradii,nwave)).T
    qscat = df['qscat'].values.reshape((nradii,nwave)).T
    qext = df['qext'].values.reshape((nradii,nwave)).T
    cos_qscat = df['cos_qscat'].values.reshape((nradii,nwave)).T

    return qext,qscat, cos_qscat, nwave, radii,wave

def get_refrind(igas,directory): 
    """
    Reads reference files with wavelength, and refractory indecies. 
    This function relies on input files being structured as a 4 column file with 
    columns: index, wavelength (micron), nn, kk 

    Parameters
    ----------
    igas : str 
        Gas name 
    directory : str 
        Directory were reference files are located. 
    """
    filename = os.path.join(directory ,igas+".refrind")
    #put skiprows=1 in loadtxt to skip first line
    idummy, wave_in, nn, kk = np.loadtxt(open(filename,'rt').readlines(), unpack=True, usecols=[0,1,2,3])#[:-1]

    return wave_in,nn,kk

def get_r_grid_w_max(r_min=1e-8, r_max=5.4239131e-2, n_radii=60):
    """
    Get spacing of radii to run Mie code

    r_min : float 
            Minimum radius to compute (cm)
    r_max : float 
            Maximum radius to compute (cm)
    n_radii : int
            Number of radii to compute 
    """

    radius = np.logspace(np.log10(r_min),np.log10(r_max),n_radii)
    rat = radius[1]/radius[0]
    rup = 2*rat / (rat+1) * radius
    dr = np.zeros(rup.shape)
    dr[1:] = rup[1:]-rup[:-1]
    dr[0] = dr[1]**2/dr[2]

    return radius, rup, dr

def get_r_grid(r_min=1e-8, n_radii=60):
    """
    Warning
    -------
    Original code from A&M code. 
    Discontinued function. See 'get_r_grid'.

    Get spacing of radii to run Mie code

    r_min : float 
        Minimum radius to compute (cm)

    n_radii : int
        Number of radii to compute 
    """
    vrat = 2.2 
    pw = 1. / 3.
    f1 = ( 2.0*vrat / ( 1.0 + vrat) )**pw
    f2 = (( 2.0 / ( 1.0 + vrat ) )**pw) * (vrat**(pw-1.0))

    radius = r_min * vrat**(np.linspace(0,n_radii-1,n_radii)/3.)
    rup = f1*radius
    dr = f2*radius

    return radius, rup, dr


def picaso_format(opd, w0, g0 ):
    df = pd.DataFrame(index=[ i for i in range(opd.shape[0]*opd.shape[1])], columns=['lvl','w','opd','w0','g0'])
    i = 0 
    LVL = []
    WV,OPD,WW0,GG0 =[],[],[],[]
    for j in range(opd.shape[0]):
           for w in range(opd.shape[1]):
                LVL +=[j+1]
                WV+=[w+1]
                OPD+=[opd[j,w]]
                WW0+=[w0[j,w]]
                GG0+=[g0[j,w]]
    df.iloc[:,0 ] = LVL
    df.iloc[:,1 ] = WV
    df.iloc[:,2 ] = OPD
    df.iloc[:,3 ] = WW0
    df.iloc[:,4 ] = GG0
    return df

def available():
    """
    Print all available gas condensates 
    """
    pvs = [i for i in dir(pvaps) if i != 'np' and '_' not in i]
    gas_p = [i for i in dir(gas_properties) if i != 'np' and '_' not in i]
    return list(np.intersect1d(gas_p, pvs))

def recommend_gas(pressure, temperature, mh, mmw, plot=False, legend='inside',**plot_kwargs):
    """
    Recommends condensate species for a users calculation. 

    Parameters
    ----------
    pressure : ndarray, list
        Pressure grid for user's pressure-temperature profile. Unit=bars
    temperature : ndarray, list 
        Temperature grid (should be on same grid as pressure input). Unit=Kelvin
    mh : float 
        Metallicity in NOT log units. Solar =1 
    mmw : float 
        Mean molecular weight of the atmosphere. Solar = 2.2 
    plot : bool, optional
        Default is False. Plots condensation curves against PT profile to 
        demonstrate why it has chose the gases it did. 
    plot_kwargs : kwargs 
        Plotting kwargs for bokeh figure

    Returns
    -------
    ndarray, ndarray
        pressure (bars), condensation temperature (Kelvin)
    """    
    if plot: 
        from bokeh.plotting import figure, show
        from bokeh.models import Legend
        from bokeh.palettes import magma   
        plot_kwargs['y_range'] = plot_kwargs.get('y_range',[1e2,1e-3])
        plot_kwargs['plot_height'] = plot_kwargs.get('plot_height',400)
        plot_kwargs['plot_width'] = plot_kwargs.get('plot_width',600)
        plot_kwargs['x_axis_label'] = plot_kwargs.get('x_axis_label','Temperature (K)')
        plot_kwargs['y_axis_label'] = plot_kwargs.get('y_axis_label','Pressure (bars)')
        plot_kwargs['y_axis_type'] = plot_kwargs.get('y_axis_type','log')        
        fig = figure(**plot_kwargs)

    all_gases = available()
    cond_ts = []
    recommend = []
    line_widths = []
    for gas_name in all_gases: #case sensitive names
        #grab p,t from eddysed
        cond_p,t = condensation_t(gas_name, mh, mmw)
        cond_ts +=[t]

        interp_cond_t = np.interp(pressure,cond_p,t)

        diff_curve = interp_cond_t - temperature

        if ((len(diff_curve[diff_curve>0]) > 0) & (len(diff_curve[diff_curve<0]) > 0)):
            recommend += [gas_name]
            line_widths +=[5]
        else: 
            line_widths +=[1]        

    if plot: 
        legend_it = []
        ngas = len(all_gases)
        cols = magma(ngas)
        if legend is 'inside':
            fig.line(temperature,pressure, legend_label='User',color='black',line_width=5,line_dash='dashed')
            for i in range(ngas):

                fig.line(cond_ts[i],cond_p, legend_label=all_gases[i],color=cols[i],line_width=line_widths[i])
        else:
            f = fig.line(temperature,pressure, color='black',line_width=5,line_dash='dashed')
            legend_it.append(('input profile', [f]))
            for i in range(ngas):

                f = fig.line(cond_ts[i],cond_p ,color=cols[i],line_width=line_widths[i])
                legend_it.append((all_gases[i], [f]))

        if legend is 'outside':
            legend = Legend(items=legend_it, location=(0, 0))
            legend.click_policy="mute"
            fig.add_layout(legend, 'right')   
        
        plot_format(fig)
        show(fig) 

    return recommend 

def condensation_t(gas_name, mh, mmw, pressure =  np.logspace(-6, 2, 20)):
    """
    Find condensation curve for any planet given a pressure. These are computed 
    based on pressure vapor curves defined in pvaps.py. 

    Default is to compute condensation temperature on a pressure grid 

    Parameters
    ----------
    gas_name : str 
        Name of gas, which is case sensitive. See print_available to see which 
        gases are available. 
    mh : float 
        Metallicity in NOT log units. Solar =1 
    mmw : float 
        Mean molecular weight of the atmosphere. Solar = 2.2 
    pressure : ndarray, list, float, optional 
        Grid of pressures (bars) to compute condensation temperatures on. 
        Default = np.logspace(-3,2,20)

    Returns
    -------
    ndarray, ndarray
        pressure (bars), condensation temperature (Kelvin)
    """
    if isinstance(pressure,(float,int)):
        pressure = [pressure]
    temps = []
    for p in pressure: 
        temp = optimize.root_scalar(find_cond_t, 
                        bracket=[10, 10000], method='brentq', 
                        args=(p, mh, mmw, gas_name))
        temps += [temp.root]
    return np.array(pressure), np.array(temps)

def hot_jupiter():
    directory = os.path.join(os.path.dirname(__file__), "reference",
                                   "hj.pt")

    df = pd.read_csv(directory,delim_whitespace=True, usecols=[1,2,3],
                  names = ['pressure','temperature','kz'],skiprows=1)
    df.loc[df['pressure']>12.8,'temperature'] = np.linspace(1822,2100,df.loc[df['pressure']>12.8].shape[0])
    return df

def brown_dwarf(): 
    directory = os.path.join(os.path.dirname(__file__), "reference",
                                   "t1000g100nc_m0.0.dat")

    df  = pd.read_csv(directory,skiprows=1,delim_whitespace=True,
                 header=None, usecols=[1,2,3],
                 names=['pressure','temperature','chf'])
    return df

