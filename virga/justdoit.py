import astropy.constants as c
import astropy.units as u
import pandas as pd
import numpy as np
import os
from scipy import optimize 
from pathlib import Path

from .root_functions import advdiff, vfall,vfall_find_root,qvs_below_model, find_cond_t, solve_force_balance
from .calc_mie import fort_mie_calc, calc_new_mieff, calc_new_mieff_optool
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
    
    # if k0 is not left as default parameter and r_mon is prescribed, print warning
    if ((atmo.k0>0) or (atmo.k0<0)) and (atmo.r_mon is not None):
        print(f"""WARNING: You have prescribed the value k0 when calling VIRGA (k0 = {atmo.k0}). If r_mon is prescribed instead of N_mon,
         this means the vfall function may have weird transitions around r=r_mon. Proceed with caution, or to avoid this, leave
         k0 blank and it will be calculated by VIRGA.""")

    # warn user if they try to use anything other than default og_vfall with aggregates, and terminate code (this is still under development)
    if((atmo.aggregates==True) and (og_vfall==False)):
        print("WARNING: If using aggregates, you need to set og_vfall=True (the default - the other method is still under development).")
        exit(1)

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
        gas_mw[i], gas_mmr[i], rho_p[i] = run_gas(mmw, mh=mh, gas_mmr=atmo.gas_mmr[igas])

        #Get mie files that are already saved in directory
        #eventually we will replace this with nice database 

        qext_gas, qscat_gas, cos_qscat_gas, nwave, radius,wave_in = get_mie(igas,directory, atmo.aggregates, atmo.Df)

        if i==0: 
            nradii = len(radius) # work our how many radii were in the .mieff file
            rmin = np.min(radius) # find the min radius
            rmax = np.max(radius) # find the max radius

            # work out if the radii in the .mieff file were created on a log-spaced grid (the default) or linearly-spaced grid
            if (((radius[2]-radius[1]) - (radius[1]-radius[0])) < 0.0000000001): # if the spacing of the mean radii for bins 0, 1 and 2 are constant...
                logspace=False # ...the grid was created with linear spacing
                log_radii=0 # save as a scalar variable for dict
            else: # if the spacing is not constant...
                logspace=True # ...the grid was created with log spacing
                log_radii=1 # save as a scalar variable for dict

            # use the get_r_grid() function to recreate the same grid that was used to create the .mieff file -- this allows us to find the bin widths (dr) for use in calc_optics() function
            radius, bin_min, bin_max, dr = get_r_grid(rmin, rmax, nradii, logspace)
            
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
        #here atmo.param describes the parameterization used for the variable fsed methodology
        if atmo.param == 'exp': 
            #the formalism of this is detailed in Rooney et al. 2021
            atmo.b = 6 * atmo.b * H # using constant scale-height in fsed
            fsed_in = (atmo.fsed-atmo.eps) 
        elif atmo.param == 'const':
            fsed_in = atmo.fsed
        qc, qt, rg, reff, ndz, qc_path, mixl, z_cld = eddysed(atmo.t_level, atmo.p_level, atmo.t_layer, atmo.p_layer, 
                                             condensibles, gas_mw, gas_mmr, rho_p , mmw, 
                                             atmo.g, atmo.kz, atmo.mixl, 
                                             fsed_in,
                                             atmo.b, atmo.eps, atmo.scale_h, atmo.z_top, atmo.z_alpha, min(atmo.z), atmo.param,
                                             mh, atmo.sig, rmin, nradii, radius,
                                             atmo.d_molecule,atmo.eps_k,atmo.c_p_factor,
                                             atmo.aggregates, atmo.Df, atmo.N_mon, atmo.r_mon, atmo.k0,
                                             og_vfall, supsat=atmo.supsat,verbose=atmo.verbose,do_virtual=do_virtual)
        pres_out = atmo.p_layer
        temp_out = atmo.t_layer
        z_out = atmo.z

    
    #   run new, direct solver
    else:
        fsed_in = atmo.fsed
        z_cld = None #temporary fix 
        qc, qt, rg, reff, ndz, qc_path, pres_out, temp_out, z_out,mixl = direct_solver(atmo.t_layer, atmo.p_layer,
                                             condensibles, gas_mw, gas_mmr, rho_p , mmw, 
                                             atmo.g, atmo.kz, atmo.fsed, mh,atmo.sig, radius, 
                                             atmo.d_molecule,atmo.eps_k,atmo.c_p_factor,
                                             atmo.aggregates,atmo.Df,atmo.N_mon,atmo.r_mon,atmo.k0, direct_tol, refine_TP, og_vfall, analytical_rg)

            
    #Finally, calculate spectrally-resolved profiles of optical depth, single-scattering
    #albedo, and asymmetry parameter.    
    opd, w0, g0, opd_gas = calc_optics(nwave, qc, qt, rg, reff, ndz,radius,
                                       dr,qext, qscat,cos_qscat,atmo.sig, rmin, rmax, verbose=atmo.verbose)

    if as_dict:
        if atmo.param == 'exp':
            fsed_out = fsed_in * np.exp((atmo.z - atmo.z_alpha) / atmo.b ) + atmo.eps
        else: 
            fsed_out = fsed_in 
        return create_dict(qc, qt, rg, reff, ndz,opd, w0, g0, 
                           opd_gas,wave_in, pres_out, temp_out, condensibles,
                           mh,mmw, fsed_out, atmo.sig, nradii,rmin, rmax, log_radii, z_out, atmo.dz_layer, 
                           mixl, atmo.kz, atmo.scale_h, z_cld) 
    else:
        return opd, w0, g0

def create_dict(qc, qt, rg, reff, ndz,opd, w0, g0, opd_gas,wave,pressure,temperature, gas_names,
    mh,mmw,fsed,sig,nrad,rmin,rmax,log_radii,z, dz_layer, mixl, kz, scale_h, z_cld):
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
        "scalar_inputs": {'mh':mh, 'mmw':mmw,'sig':sig,'nrad':nrad,'rmin':rmin,'rmax':rmax,'log_radii':log_radii},
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

def calc_optics(nwave, qc, qt, rg, reff, ndz,radius,dr,qext, qscat,cos_qscat,sig, rmin, rmax, verbose=False):
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
        Particle radius bin centers from the grid (cm)
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
    rmin: float
        Minimum particle radius bin center from the grid (cm)
    rmax: float
        Maximum particle radius bin center from the grid (cm)
    verbose: bool 
        print out warnings or not


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
    warning=''
    for iz in range(nz):
        for igas in range(ngas):
            # Optical depth for conservative geometric scatterers 
            if ndz[iz,igas] > 0:
                
                # precise warning for when particles are either above or below the grid defined by the .mieff files
                if rg[iz,igas] < rmin: # if the radius of the particle found by VIRGA is smaller than rmin it is "off the grid". We assume this because even if it was at the bottom of the smallest bin e.g. r_g = (r_min - dr/2), half the distribution would not be considered by the number density, so we need to make the grid bigger to account for this portion of smaller particles  
                    warning0 = f'Take caution in analyzing results. Particle sizes were predicted by eddysed() that were smaller than the minimum radius in the .mieff file ({rmin} cm). The optics and number densities for these particles will therefore not be correct. This can be solved by recreating the .mieff grids with a smaller r_min. The errors occurred at the following pressure layers:'
                    warning+='\nParticles of radius {0} cm were found (where rmin from the mieff file is {1} cm) for gas {2} in the {3}th altitude layer'.format(str(rg[iz,igas]),str(rmin),str(igas),str(iz))

                if rg[iz,igas] > rmax: # if the radius of the particle found by VIRGA is larger than rmax it is "off the grid". We assume this because even if it was at the top of the largest bin e.g. r_g = (r_max + dr/2), half the distribution would not be considered by the number density, so we need to make the grid bigger to account for this portion of larger particles
                    warning0 = f'Take caution in analyzing results. Particle sizes were predicted by eddysed() that were larger than the maximum radius in the .mieff file ({rmax} cm). The optics and number densities for these particles will therefore not be correct. This can be solved by recreating the .mieff grids with a larger r_max. The errors occurred at the following pressure layers:'
                    warning+='\nParticles of radius {0} cm were found (where rmax from the mieff file is {1} cm) for gas {2} in the {3}th altitude layer'.format(str(rg[iz,igas]),str(rmax),str(igas),str(iz))

                r2 = rg[iz,igas]**2 * np.exp( 2*np.log( sig)**2 )
                opd_layer[iz,igas] = 2.*PI*r2*ndz[iz,igas]

                #  Calculate normalization factor (forces lognormal sum = 1.0)
                rsig = sig #the log normal particle size distribution 
                norm = 0.
                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / ( np.sqrt(2.*PI)*rr*np.log(rsig) )
                    arg2 = -np.log( rr/rg[iz,igas] )**2 / ( 2*np.log(rsig)**2 )
                    norm = norm + arg1*np.exp( arg2 )
                    #print (rr, rg[iz,igas],rsig,arg1,arg2)

                # normalization
                norm = ndz[iz,igas] / norm #number density distribution

                for irad in range(nrad):
                    rr = radius[irad]
                    arg1 = dr[irad] / ( np.sqrt(2.*PI)*np.log(rsig) ) # log normal distribution is the rsig
                    arg2 = -np.log( rr/rg[iz,igas] )**2 / ( 2*np.log(rsig)**2 )
                    pir2ndz = norm*PI*rr*arg1*np.exp( arg2 )         
                    for iwave in range(nwave): 
                        scat_gas[iz,iwave,igas] = scat_gas[iz,iwave,igas]+qscat[iwave,irad,igas]*pir2ndz
                        ext_gas[iz,iwave,igas] = ext_gas[iz,iwave,igas]+qext[iwave,irad,igas]*pir2ndz
                        cqs_gas[iz,iwave,igas] = cqs_gas[iz,iwave,igas]+cos_qscat[iwave,irad,igas]*pir2ndz

                    #TO DO ADD IN CLOUD SUBLAYER KLUGE LATER 
    
    for igas in range(ngas):
        for iz in range(nz-1,-1,-1):

            if np.sum(ext_gas[iz,:,igas]) > 0:
                ibot = iz
                break
            if iz == 0:
                ibot=0
        #print(igas,ibot)
        if ibot >= nz -2:
            print("Not doing sublayer as cloud deck at the bottom of pressure grid")
            
        else:
            opd_layer[ibot+1,igas] = opd_layer[ibot,igas]*0.1
            scat_gas[ibot+1,:,igas] = scat_gas[ibot,:,igas]*0.1
            ext_gas[ibot+1,:,igas] = ext_gas[ibot,:,igas]*0.1
            cqs_gas[ibot+1,:,igas] = cqs_gas[ibot,:,igas]*0.1
            opd_layer[ibot+2,igas] = opd_layer[ibot,igas]*0.05
            scat_gas[ibot+2,:,igas] = scat_gas[ibot,:,igas]*0.05
            ext_gas[ibot+2,:,igas] = ext_gas[ibot,:,igas]*0.05
            cqs_gas[ibot+2,:,igas] = cqs_gas[ibot,:,igas]*0.05
            opd_layer[ibot+3,igas] = opd_layer[ibot,igas]*0.01
            scat_gas[ibot+3,:,igas] = scat_gas[ibot,:,igas]*0.01
            ext_gas[ibot+3,:,igas] = ext_gas[ibot,:,igas]*0.01
            cqs_gas[ibot+3,:,igas] = cqs_gas[ibot,:,igas]*0.01
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
    if ((warning!='') & (verbose)): print(warning0+warning+' Turn off warnings by setting verbose=False.')
    return opd, w0, g0, opd_gas

def calc_optics_user_r_dist(wave_in, ndz, 
    radius, radius_unit, r_distribution, 
    qext, qscat ,cos_qscat,  verbose=False):
    """
    Calculate spectrally-resolved profiles of optical depth, single-scattering
    albedo, and asymmetry parameter for a user-input particle radius distribution

    Parameters
    ----------
    wave_in : ndarray
        your wavelength grid in microns
    ndz : float
        Column density of total particle concentration (#/cm^2) 
            Note: set to whatever, it's your free knob 
            ---- this does not directly translate to something physical because it's for all particles in your slab
            May have to use values of 1e8 or so
    radius : ndarray
        Radius bin values - the range of particle sizes of interest. Maybe measured in the lab, 
        Ensure radius_unit is specified 
    radius_unit : astropy.unit.Units
        Astropy compatible unit
    qscat : ndarray
        Scattering efficiency
    qext : ndarray
        Extinction efficiency
    cos_qscat : ndarray
        qscat-weighted <cos (scattering angle)>
    r_distribution : ndarray
        the radius distribution in each bin. Maybe measured from the lab, generated from microphysics, etc.
        Should integrate to 1. 
    verbose: bool 
        print out warnings or not


    Returns
    -------
    opd : ndarray 
        extinction optical depth due to all condensates in layer
    w0 : ndarray 
        single scattering albedo
    g0 : ndarray 
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    """

    radius = (radius*radius_unit).to(u.cm)
    radius = radius.value
    
    wavenumber_grid = 1e4/wave_in
    wavenumber_grid = np.array([item[0] for item in wavenumber_grid])
    nwave = len(wavenumber_grid)
    PI=np.pi
    nrad = len(radius) ## where radius is the radius grid of the particle size distribution
    
    scat= np.zeros((nwave))
    ext = np.zeros((nwave))
    cqs = np.zeros((nwave))
    
    opd = np.zeros((nwave))
    w0 = np.zeros((nwave))
    g0 = np.zeros((nwave))
    
    opd_scat = 0.
    opd_ext = 0.
    cos_qs = 0.
                    
        #  Calculate normalization factor 
    for irad in range(nrad):
            rr = radius[irad] # the get the radius at each grid point, this is in nanometers 
    
            each_r_bin = ndz * (r_distribution[irad]) # weight the radius bin by the distribution 
            pir2ndz = PI * rr**2 * each_r_bin # find the weighted cross section
            
            for iwave in range(nwave): 
                scat[iwave] = scat[iwave] + qscat[iwave,irad]*pir2ndz 
                ext[iwave] = ext[iwave] + qext[iwave,irad]*pir2ndz
                cqs[iwave] = cqs[iwave] + cos_qscat[iwave,irad]*pir2ndz
                
                
                    # calculate the spectral optical depth profile etc
    for iwave in range(nwave): 
            opd_scat = 0.
            opd_ext = 0.
            cos_qs = 0.
            
            opd_scat = opd_scat + scat[iwave]
            opd_ext = opd_ext + ext[iwave]
            cos_qs = cos_qs + cqs[iwave]
   
                    
            if( opd_scat > 0. ):
                            opd[iwave] = opd_ext 
                            w0[iwave] = opd_scat / opd_ext
                            g0[iwave] = cos_qs / opd_scat
                    
    return opd, w0, g0, wavenumber_grid

def eddysed(t_top, p_top,t_mid, p_mid, condensibles, 
    gas_mw, gas_mmr,rho_p,mw_atmos,gravity, kz,mixl,
    fsed, b, eps, scale_h, z_top, z_alpha, z_min, param,
    mh,sig, rmin, nrad, radius, d_molecule,eps_k,c_p_factor,
    aggregates, Df, N_mon, r_mon, k0, og_vfall=True,do_virtual=True, supsat=0, verbose=False):
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
    scale_h : float 
        Scale height of the atmosphere
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
    rmin : float 
        Minium radius on grid (cm)
    nrad : int 
        Number of radii on Mie grid
    radius : ndarray
        Particle radius bin centers from the grid (cm)
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
    aggregates : bool, optional
        set to 'True' if you want fractal aggregates, keep at default 'False' for spherical cloud particles
    Df : float, optional
        only used if aggregate = True.
        The fractal dimension of an aggregate particle. 
        Low Df are highly fractal long, lacy chains; large Df are more compact. Df = 3 is a perfect compact sphere.
    N_mon : float, optional
        only used if aggregate = True. 
        The number of monomers that make up the aggregate. Either this OR r_mon should be provided (but not both).
    r_mon : float, optional (units: cm)
        only used if aggregate = True. 
        The size of the monomer radii (sub-particles) that make up the aggregate. Either this OR N_mon should be 
        provided (but not both).
    k0 : float, optional (units: None)
        only used if aggregate = True. 
        Default = 0, where it will then be calculated in the vfall equation using Tazaki (2021) Eq 2. k0 can also be prescribed by user, 
        but with a warning that when r_mon is fixed, unless d_f = 1 at r= r_mon, the dynamics may not be consistent between the boundary 
        when spheres grow large enough to become aggregates (this applies only when r_mon is fixed. If N_mon is fixed instead, any value of k0 is fine).
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
            z_cld=None
            qvs_factor = (supsat+1)*gas_mw[i]/mw_atmos
            get_pvap = getattr(pvaps, igas)
            if igas in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
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
                    z_base = z_bot + scale_h[-1] * np.log( p_bot/p_base ) 
                    
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
                        sig,mh, rmin, nrad, radius, d_molecule,eps_k,c_p_factor, #all scalars
                        og_vfall, z_cld, aggregates, Df, N_mon, r_mon, k0
                    )

        z_cld=None
        for iz in range(nz-1,-1,-1): #goes from BOA to TOA

            qc[iz,i], qt[iz,i], rg[iz,i], reff[iz,i],ndz[iz,i],q_below, z_cld, fsed_layer[iz,i]  = layer( igas, rho_p[i], 
                #t,p layers, then t.p levels below and above
                t_mid[iz], p_mid[iz], t_top[iz], t_top[iz+1], p_top[iz], p_top[iz+1],
                kz[iz], mixl[iz], gravity, mw_atmos, gas_mw[i], q_below,  
                supsat, fsed, b, eps, z_top[iz], z_top[iz+1], z_alpha, z_min, param,
                sig,mh, rmin, nrad, radius, d_molecule,eps_k,c_p_factor, #all scalars
                og_vfall, z_cld, aggregates, Df, N_mon, r_mon, k0
            )

            qc_path[i] = (qc_path[i] + qc[iz,i]*
                            ( p_top[iz+1] - p_top[iz] ) / gravity)
        z_cld_out[i] = z_cld

    return qc, qt, rg, reff, ndz, qc_path,mixl, z_cld_out

def layer(gas_name,rho_p, t_layer, p_layer, t_top, t_bot, p_top, p_bot,
    kz, mixl, gravity, mw_atmos, gas_mw, q_below,
    supsat, fsed, b, eps, z_top, z_bot, z_alpha, z_min, param,
    sig,mh, rmin, nrad, radius, d_molecule,eps_k,c_p_factor,
    og_vfall, z_cld, aggregates, Df, N_mon, r_mon, k0):
    """
    Calculate layer condensate properties by iterating on optical depth
    in one model layer (converging on optical depth over sublayers)

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
    radius : ndarray
        Particle radius bin centers from the grid (cm)
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
    aggregates : bool, optional
        set to 'True' if you want fractal aggregates, keep at default 'False' for spherical cloud particles
    Df : float, optional
        only used if aggregate = True.
        The fractal dimension of an aggregate particle. 
        Low Df are highly fractal long, lacy chains; large Df are more compact. Df = 3 is a perfect compact sphere.
    N_mon : float, optional
        only used if aggregate = True. 
        The number of monomers that make up the aggregate. Either this OR r_mon should be provided (but not both).
    k0 : float, optional (units: None)
        only used if aggregate = True. 
        Default = 0, where it will then be calculated in the vfall equation using Tazaki (2021) Eq 2. k0 can also be prescribed by user, 
        but with a warning that when r_mon is fixed, unless d_f = 1 at r= r_mon, the dynamics may not be consistent between the boundary 
        when spheres grow large enough to become aggregates (this applies only when r_mon is fixed. If N_mon is fixed instead, any value of k0 is fine).
    r_mon : float, optional (units: cm)
        only used if aggregate = True. 
        The size of the monomer radii (sub-particles) that make up the aggregate. Either this OR N_mon should be 
        provided (but not both).

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

        #SUBLAYER 
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
                        sig,mh, rmin, nrad, radius, aggregates, Df, N_mon, r_mon, k0, og_vfall, z_cld)


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
    sig, mh, rmin, nrad, radius, aggregates, Df, N_mon, r_mon, k0, og_vfall=True,z_cld=None):
    """
    Calculate condensate number density and effective radius for a layer,
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
    radius : ndarray
        Particle radius bin centers from the grid (cm)
    aggregates : bool, optional
        set to 'True' if you want fractal aggregates, keep at default 'False' for spherical cloud particles
    Df : float, optional
        only used if aggregate = True.
        The fractal dimension of an aggregate particle. 
        Low Df are highly fractal long, lacy chains; large Df are more compact. Df = 3 is a perfect compact sphere.
    N_mon : float, optional
        only used if aggregate = True. 
        The number of monomers that make up the aggregate. Either this OR r_mon should be provided (but not both).
    r_mon : float, optional (units: cm)
        only used if aggregate = True. 
        The size of the monomer radii (sub-particles) that make up the aggregate. Either this OR N_mon should be 
        provided (but not both).
    k0 : float, optional (units: None)
        only used if aggregate = True. 
        Default = 0, where it will then be calculated in the vfall equation using Tazaki (2021) Eq 2. k0 can also be prescribed by user, 
        but with a warning that when r_mon is fixed, unless d_f = 1 at r= r_mon, the dynamics may not be consistent between the boundary 
        when spheres grow large enough to become aggregates (this applies only when r_mon is fixed. If N_mon is fixed instead, any value of k0 is fine).
    og_vfall : bool , optional
        optional, default = True. True does the original fall velocity calculation. 
        False does the updated one which runs a tad slower but is more consistent.
        The main effect of turning on False is particle sizes in the upper atmosphere 
        that are slightly bigger.

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
    if gas_name in ['Mg2SiO4','CaTiO3','CaAl12O19','FakeHaze','H2SO4','KhareHaze','SteamHaze300K','SteamHaze400K']:
        pvap = get_pvap(t_layer, p_layer,mh=mh)
    else:
        pvap = get_pvap(t_layer,mh=mh)

    fs = supsat + 1 

    #   atmospheric density (g/cm^3)
    rho_atmos = p_layer / ( r_atmos * t_layer )    

    #   mass mixing ratio of saturated vapor (g/g)
    qvs = fs*pvap / ( (r_cloud) * t_layer ) / rho_atmos

    #   --------------------------------------------------------------------
    #   Layer is cloud free -neb-
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
        if isinstance(z_cld,type(None)):
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
        if param == "const":
            qt_top = qvs + (q_below - qvs) * np.exp(-fsed * dz_layer / mixl)
        elif param == "exp":
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
        #   NOTE FROM MGL: For aggregates, this finds the COMPACT <rw> (the radius of a sphere of equivalent mass to the aggregate). The vfall
        #   function receives a compact radius as an input as well as a shape type (e.g. d_f = 1.7), and converts the compact radius into
        #   the radius of gyration, finds vfall, and then returns the COMPACT radius that had a radius of gyration that achieved the
        #   desired vfall. This is so that the compact radii still line up with the .mieff grid, which is also stored by COMPACT radius,
        #   and because it's easier to compare aggregates if they all have the same mass (because aggregates with the same compact radius
        #   have the same mass, whatever shape they have) at the same grid points (rather than having to convert backwards from R_gyr). The 
        #   equivalent radii of gyration can be found using the "convert_rg_to_R_gyr" function in the analysis tools, if required.

        # Note: MGL and SEM have decreased the range of these limits to start the initial search much narrower to improve 
        # stability of the iterative solver for aggregates. The initial search range will be increased if a solution is not found
        # on the first attempt anyway (using rhi = rhi*10 in the 'try' loop below). Previously, spheres were allowed
        # to be smaller than gas atoms, but we have set this as a lower limit and printed a warning if VIRGA can't find a solution
        # for spheres -- see 3rd 'Exception' below)
        
        #   range of particle radii to search (in cm)
        rlo = 1.e-8 # the minimum particle size to search is 0.1 nm for spheres 
        if (aggregates==True):       
            rhi = 1.e-7  # for aggregates, begin the search with a small initial maximum particle size and only make the search wider if no solution is found (to ensure that the lowest value is taken in cases with multiple roots/solutions -- see Moran & Lodge 2025)
        else:
            rhi = 1.0 # for spheres, we can begin the search with quite a wide range as multiple solutions are not expected. The initial maximum particle size to search is 1 cm (but this limit will be increased if no solution is found)

        
        
        #   precision of vfall solution (cm/s)
        find_root = True
        while find_root:
            try:
                if og_vfall:
                    rw_temp = optimize.root_scalar(vfall_find_root, bracket=[rlo, rhi], method='brentq', 
                            args=(gravity,mw_atmos,mfp,visc,t_layer,p_layer, rho_p,w_convect, aggregates, Df, N_mon, r_mon, k0))
                else:
                    rw_temp = solve_force_balance("rw", w_convect, gravity, mw_atmos, mfp,
                                                    visc, t_layer, p_layer, rho_p, rlo, rhi)
                find_root = False
            except ValueError:
                #rlo = rlo/10 # MGL and SEM have commented this out, because you could never form particles smaller than atoms (10^-8 cm)
                rhi = rhi*10
                
                # warning to user that the iterative solver has not found a solution (if you prescribe N_mon or r_mon, there are
                # some situations where a solution is just not physically possible i.e. if fluffy aggregates are being lofted high
                # up by a strong Kzz value, no matter how large they become in radius, they may not ever have a fall velocity that
                # balances this against w_convect)


                # if tested particle is larger than Jupiter (!), raise a warning that no solution could be found

                if aggregates:
                    if N_mon is not None:
                        if rhi>1e10 : raise Exception(f"Warning: Could not find a solution that allows particles of this N_mon ({N_mon}) to balance w_convect for {Df}.\
                                                      \nPlease try using a smaller value of N_mon, or decrease Kzz.")
                    else:
                        if rhi>1e10 : raise Exception(f"Warning: Could not find a solution that allows particles of this r_mon ({r_mon} cm) to balance w_convect for {Df}.\
                                                      \nPlease try using a larger value of r_mon, or decrease Kzz.")
                else:
                    if rhi>1e10 : raise Exception(f"Warning: Could not find a physical solution that balances w_convect (Kzz is so low \
                                                  \n that particles would need to be smaller than gas atoms to stay in the pressure layer). \
                                                  \n Please try using a higher value Kzz.")

        #fall velocity particle radius 
        if og_vfall: rw_layer = rw_temp.root
        else: rw_layer = rw_temp

        # MGL NOTE: In this section, we try to link r_w (the radius where convective upwards velocity = downwards velocity, which is found above) to the r_g (geometric
        # mean radius), which we will calculate the optical properties from. We assume the same constant 'alpha' links spheres and aggregates of any shape for fair comparison.
        # The original A+M code suggests a narrow range of particle radii to determine alpha from, but because v_fall in VIRGA is more complex than vfall in the A+M model,
        # the original version of VIRGA found an average alpha using the entire .mieff grid. MGL ran tests to see if the A+M method range would give better results for aggregates, 
        # but there was little difference so the code is unchanged here. The method is as follows:
        #
        # 1) Find vfall values for each of the radii in the .mieff grid, using the vfall equation for spheres.
        # 2) Calculate the constant of proportionality (alpha) that links this array of vfall values to the convective velocity x ratio of("grid radius" / "radius of particles with vfall equal to w_convect in this layer"), using the properties (pressure, temp, gravity) in this layer exclusively.
        # Note: This is a way of seeing how particle sizes would scale if the conditions were the same everywhere. Alpha is a single value, and often simply = 1.
        # 3) Use this alpha to find r_g (geometric mean radii) and droplet effective radius in the layer, assuming a lognormal distribution.
        #  

            #   geometric std dev of lognormal size distribution
        lnsig2 = 0.5*np.log( sig )**2
        #   sigma floor for the purpose of alpha calculation
        sig_alpha = np.max( [1.1, sig] )    

        #   find alpha for power law fit vf = w(r/rw)^alpha
        def pow_law(r, alpha):
            return np.log(w_convect) + alpha * np.log (r / rw_layer_spheres) # use spherical version of r_w, calculated below

        # find value of r_w that would exist for spherical particles -- this is needed in the calculation of alpha, no matter what the particle shape is, because we calcualte alpha based on the spherical version

        find_root = True
        while find_root:
            try:
                rw_temp_spheres = optimize.root_scalar(vfall_find_root, bracket=[rlo, rhi], method='brentq', 
                        args=(gravity,mw_atmos,mfp,visc,t_layer,p_layer, rho_p,w_convect, False, 0, 0, 0, 0)) # use aggregates = False so that we are just using the spherical version of v_fall
                find_root = False
            except ValueError:
                #rlo = rlo/10 # MGL and SEM have commented this out, because you could never form particles smaller than atoms (10^-8 cm)
                rhi = rhi*10

                if rhi>1e10 : raise Exception(f"Warning: Could not find a physical solution for SPHERES (needed in the alpha calculation) \
                                                \n that balances w_convect (Kzz is so low that particles would need to be smaller than gas \
                                                \n atoms to stay in the pressure layer). Please try using a higher value Kzz.")
        rw_layer_spheres = rw_temp_spheres.root


        # calculate vfall for each radius in the .mieff file, assuming spherical particles
        vfall_temp = []
        for j in range(len(radius)):
            if og_vfall:
                vfall_temp.append(vfall(radius[j], gravity, mw_atmos, mfp, visc, t_layer, p_layer, rho_p, aggregates=False, Df=0, N_mon=0, r_mon=0, k0=0)) # this calculates vfall for each of the radii in the .mieff grid for SPHERES. We then assume the same link between r_w and r_g holds for all shapes fractal dimensions as it does for spheres. This is to avoid issues with v_fall being much more complex than the original A+M model, and to make a fairer comparison between different shapes. 
            else:
                vlo = 1e0; vhi = 1e6
                find_root = True
                while find_root:
                    try:
                        vfall_temp.append(solve_force_balance("vfall", radius[j], gravity, mw_atmos, 
                            mfp, visc, t_layer, p_layer, rho_p, vlo, vhi))
                        find_root = False
                    except ValueError:
                        vlo = vlo/10
                        vhi = vhi*10

        # determine alpha, assuming spherical particles
        pars, cov = optimize.curve_fit(f=pow_law, xdata=radius, ydata=np.log(vfall_temp).ravel(), p0=[0], # this code finds alpha (a constant of proportionality) for each of the radius values in the .mieff grid, assuming that they scale with v_fall in a power law
                            bounds=(-np.inf, np.inf))
        alpha = pars[0]

        #   fsed at middle of layer 
        if param == 'exp':
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
                    param='const', verbose=False, supsat=0, gas_mmr=None,
                    aggregates=False, Df=None, N_mon=None, r_mon=None,k0=0):
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
        aggregates : bool, optional
            set to 'True' if you want fractal aggregates, keep at default 'False' for spherical cloud particles
        Df : float, optional
            only used if aggregate = True.
            The fractal dimension of an aggregate particle. 
            Low Df are highly fractal long, lacy chains; large Df are more compact. Df = 3 is a perfect compact sphere.
        N_mon : float, optional
            only used if aggregate = True. 
            The number of monomers that make up the aggregate. Either this OR r_mon should be provided (but not both).
        r_mon : float, optional (units: cm)
            only used if aggregate = True. 
            The size of the monomer radii (sub-particles) that make up the aggregate. Either this OR N_mon should be 
            provided (but not both).
        k0 : float, optional (units: None)
            only used if aggregate = True. 
            Default = 0, where it will then be calculated in the vfall equation using Tazaki (2021) Eq 2. k0 can also be prescribed by user, 
            but with a warning that when r_mon is fixed, unless d_f = 1 at r= r_mon, the dynamics may not be consistent between the boundary 
            when spheres grow large enough to become aggregates (this applies only when r_mon is fixed. If N_mon is fixed instead, any value of k0 is fine).

    
        """
        if isinstance(condensibles, str):
            self.condensibles = [condensibles]
        else:
            self.condensibles = condensibles
        self.mh = mh
        self.mmw = mmw
        self.fsed = fsed
        self.b = b
        self.sig = sig
        self.param = param
        self.eps = eps
        self.verbose = verbose 
        self.aggregates=aggregates
        self.Df=Df
        self.N_mon=N_mon
        self.r_mon=r_mon
        self.k0=k0
        #grab constants
        self.constants()
        self.supsat = supsat
        if isinstance(gas_mmr, type(None)):
            self.gas_mmr = {igas:None for igas in self.condensibles}
        else: 
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
        r"""
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
            skiprows=3, sep=r'\s+', header=None, names=["ind","pressure","temperature","kz"]
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
                if self.verbose: print("""Convective overshoot was turned on. The convective heat flux 
                    has been adjusted such that it is not allowed to decrease more than {0} 
                    the pressure. This number is set with the convective_overshoot parameter. 
                    It can be disabled with convective_overshoot=None. To turn
                    off these messages set verbose=False in Atmosphere""".format(convective_overshoot)) 

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

def calc_mie_db(gas_name, virga_dir, dir_out, optool_dir=None, rmin = 1e-8, rmax = 5.4239131e-2, nradii = 60, logspace=True, aggregates=False, Df=None, N_mon=None, r_mon=None, k0=0, fort_calc_mie = False):
    """
    Function that calculations new Mie database using MiePython (for spherical particles) or OPTOOL (for aggregates)
    Parameters
    ----------
    gas_name : list, str
        List of names of gasses. Or a single gas name. 
        See pyeddy.available() to see which ones are currently available. 
    virga_dir: str 
        Directory where you store the VIRGA refractive index files (.refind format).
    dir_out: str 
        Directory where you want to store Mie parameter files. Will be stored as gas_name.Mieff. 
        BEWARE FILE OVERWRITES. 
    optool_dir: str 
        Directory where you store optool refractive index files (.lnk format).
    rmin : float
        (Default=1e-8) Units of cm. The minimum radius to compute Mie parameters for. 
        Usually 0.001 microns is small enough. However, if you notice your mean particle radius 
        is on the low end, you may compute your grid to even lower particle sizes. 
    rmax : float
        (Default=5.4239131e-2) Units of cm. The maximum radius to compute Mie parameters for. 
        If your .mieff files take a long time to create, reduce this number. If you get a warning 
        that your particles are "off the grid" (larger than your maximum grid value), increase this number.
    nradii : int, optional
        (Default=60) number of radii points to compute grid on. 40 grid points for exoplanets/BDs
    logspace : boolean, optional
        (Default = True)
        Spaces the radii logarithmically (which they generally tend to be in the clouds) if True. Spaces them linearly if False.
    aggregates : boolean, optional
        (Default = True)
        Sets whether particles are aggregates or spheres. If true, we will calculate optical properties using OPTOOL instead of MiePython.
    Df : If aggregates = True, the fractal dimension of the aggegrate particle. a number between 1 and 3. (3 would be equal to solid spheres with radii of the outer effective radius)
    N_mon : If aggregates = True, the number of monomers that make up an aggregate. Can set directly or calculate from r_mon.
    r_mon : If aggregates = True, the monomer radius that makes up an aggregate particle (in cm). Can either set or calculate from N_mon.
    k0 : only used if aggregate = True. 
        Default = 0, where it will then be calculated in the vfall equation using Tazaki (2021) Eq 2. k0 can also be prescribed by user, 
        but with a warning that when r_mon is fixed, unless d_f = 1 at r= r_mon, the dynamics may not be consistent between the boundary 
        when spheres grow large enough to become aggregates (this applies only when r_mon is fixed. If N_mon is fixed instead, any value of k0 is fine).
    
    Returns 
    -------
    Q extinction, Q scattering,  asymmetry * Q scattering, radius grid (cm), wavelength grid (um)

    The Q "efficiency factors" are = cross section / geometric cross section of particle
    """
    
    if isinstance(gas_name,str):
        gas_name = [gas_name]
    ngas = len(gas_name)


    for i in range(len(gas_name)): 
        
        if aggregates==False:
            print('\nComputing optical properties for ' + gas_name[i] + ' using MiePython...')
            refrind_dir = virga_dir # use the VIRGA directory
        else:
            print('\nComputing optical properties for ' + gas_name[i] + ' using OPTOOL...')
            refrind_dir = optool_dir # use the optool directory

            # check whether r_mon is ever larger than the smallest particles in the grid, if it is prescribed by the user
            if r_mon is not None:
                if r_mon>rmin:

                    print("\n\n-----------------------------------------------------------------------------------------------------------------")
                    print("                                         WARNING!!!!                                         ")
                    print(f"You have prescribed a monomer size that is larger than the smallest particles in the grid:\n\n \
                            r_mon is {r_mon} cm   >   rmin is {rmin} cm) \n")
                    print("This is impossible, so for these cases, single spheres will be used, with a radius equal to the bin mean.\nThis is not a problem, but just be aware that these particles are smaller than the r_mon that you have asked for.")
                    print("------------------------------------------------------------------------------------------------------------------\n")

        print('If this function seems to be running a really long time... Check that you gave your rmin (and r_mon if using aggregates) in centimeters and that you are not making tennis balls :)')             


        #Setup up a particle size grid on first run and calculate single-particle scattering
        
        #files will be saved in `directory`
        # obtaining refractive index data for each gas
        wave_in,nn,kk = get_refrind(gas_name[i],refrind_dir, aggregates)
        nwave = len(wave_in)
        print(f'\n{nwave} wavelengths of refractive index data found for {gas_name[i]}. Grid of mean radii (in bins) to calculate extinction and scattering efficiencies for (in cm):')

        if i==0:
            # create the grid of mean radii (in bins) that we want to find the optical properties for
            radius, bin_min, bin_max, dr = get_r_grid(rmin, rmax, nradii, logspace)

            print('\n\t       min             mean             max          bin width (dr) ')
            for j in range (len(radius)):
                print(f'\t  {bin_min[j]:13.6e}   {radius[j]:13.6e}   {bin_max[j]:13.6e}      {dr[j]:13.6e}') # bins 1 -> n-1
            print('\nAverages from 6 sub-bins will be used to calculate the properties that represent the mean radius in each bin above.')

            qext_all=np.zeros(shape=(nwave,nradii,ngas))
            qscat_all = np.zeros(shape=(nwave,nradii,ngas))
            cos_qscat_all=np.zeros(shape=(nwave,nradii,ngas))

        #get extinction, scattering, and asymmetry
        #all of these are  [nwave by nradii]
  
        
        gas = gas_name[i]

        if aggregates==False: # Use MiePython to calculate the optical properties of spherical particles for each radius in the grid, using Mie theory
            qext_gas, qscat_gas, cos_qscat_gas = calc_new_mieff(wave_in, nn,kk, radius, bin_min, bin_max, fort_calc_mie = fort_calc_mie)

        else: # use OPTOOL to calculate the optical properties of aggregates for each radius in the grid, using Modified Mean Field theory   

            qext_gas, qscat_gas, cos_qscat_gas = calc_new_mieff_optool(wave_in, radius, bin_min, bin_max, gas, optool_dir, aggregates=True, Df=Df, N_mon=N_mon, r_mon=r_mon, k0=k0)
            
            # if using the OPTOOL .lnk file, the wavelengths were in reversed into ascending order (a pre-requisite of OPTOOL). Therefore, to get them back to
            # descending order, we just need to flip all arrays. Doing this means it doesn't matter whether OPTOOL or MiePython is used -- the resulting .mieff files
            # will be saved consistently in order of descending wavelength.

            wave_in = np.flipud(wave_in) # remember to reverse the wavelength array into descending order too! (currently in ascending order, from the .lnk file)
            qext_gas = np.flipud(qext_gas)
            qscat_gas = np.flipud(qscat_gas)
            cos_qscat_gas = np.flipud(cos_qscat_gas)


        #add to master matrix that contains the per gas Mie stuff
        qext_all[:,:,i], qscat_all[:,:,i], cos_qscat_all[:,:,i] = qext_gas, qscat_gas, cos_qscat_gas 

        #prepare format for old ass style
        wave = [nwave] + sum([[r]+list(wave_in) for r in radius],[])
        qscat = [nradii]  + sum([[np.nan]+list(iscat) for iscat in qscat_gas.T],[])
        qext = [np.nan]  + sum([[np.nan]+list(iext) for iext in qext_gas.T],[])
        cos_qscat = [np.nan]  + sum([[np.nan]+list(icos) for icos in cos_qscat_gas.T],[])

        if aggregates==False: # save dataframe as a standard .mieff file
            pd.DataFrame({'wave':wave,'qscat':qscat,'qext':qext,'cos_qscat':cos_qscat}).to_csv(os.path.join(dir_out,gas_name[i]+".mieff"),
                                                                                    sep=' ',
                                                                                    index=False,header=None)
            print(f'Optical properties for {gas_name[i]} have been calculated and saved as {dir_out}/{gas_name[i]}.mieff.\n')
        else: # save dataframe with a unique filename (e.g SiO2_aggregates_Df_1.2.mieff) for aggregate versions of materials

            aggregate_filename = f"{dir_out}/{gas_name[i]}_aggregates_Df_{Df:.6f}.mieff" # the name of the aggregate database file

            pd.DataFrame({'wave':wave,'qscat':qscat,'qext':qext,'cos_qscat':cos_qscat}).to_csv(aggregate_filename,
                                                                                    sep=' ',
                                                                                    index=False,header=None)
            print(f'Optical properties for {gas_name[i]} have been calculated and saved as {aggregate_filename}.\n')

    return qext_all, qscat_all, cos_qscat_all, radius,wave_in

def get_mie(gas, directory, aggregates=False, Df=None):
    """
    Get pre-calculated radius-wavelength grid of optical properties, considering the particles as either spheres (gas_name.mieff) 
    or aggregates (gas_name_aggregates_Df_XXXX.mieff), if aggregates=true and this aggregates file has been created using calc_mie_db function).
    
    """

    if aggregates==False: # load regular .mieff file of optical properties
        df = pd.read_csv(os.path.join(directory,gas+".mieff"),names=['wave','qscat','qext','cos_qscat'], sep=r'\s+')
    else: # load aggregate version of optical properties (this file must be created using the calc_mie_db function with aggregates=True first!)
        
        aggregate_filename = f"{directory}/{gas}_aggregates_Df_{Df:.6f}.mieff" # the name of the database file

        try: # check if database has been created for this particular set of MMF parameters and radii
            test_path = Path(aggregate_filename).resolve(strict=True)
        except FileNotFoundError: # aggregates database does not yet exist (for this particular gas and fractal dimension)
            print(f'File not found: {aggregate_filename}.')
            print(f'No optical database has been created for {gas} aggregates with d_f={Df} yet!')
            print('You need to create a database using the calc_mie_db() function, with the argument aggregates=True (see "aggregates" tutorial for more details).')
        else: # database file exists - load data
            print(f'Optics file found. Reading data from: {aggregate_filename}.')
            df = pd.read_csv(aggregate_filename, names=['wave','qscat','qext','cos_qscat'], sep=r'\s+')

    nwave = int( df.iloc[0,0])
    nradii = int(df.iloc[0,1])

    #get the radii (all the rows where there the last three rows are nans)
    radii = df.loc[np.isnan(df['qscat'])]['wave'].values

    df = df.dropna()

    assert len(radii) == nradii , "Number of radii specified in header is not the same as number of radii."
    assert nwave*nradii == df.shape[0] , "Number of wavelength specified in header is not the same as number of waves in file"

    # check if incoming wavegrid is in correct order
    sub_array = df['wave'].values[:196]  # Extract the first 196 values
    is_ascending = np.all(np.diff(sub_array) >= 0) # check if going from short to long wavelength

    if is_ascending == False:
        flipped_wave = np.flip(df['wave'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_qscat = np.flip(df['qscat'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_qext = np.flip(df['qext'].values.reshape(nradii, -1, nwave), axis=2).flatten()
        flipped_cos_qscat = np.flip(df['cos_qscat'].values.reshape(nradii, -1, nwave), axis=2).flatten()

        df['wave'] = flipped_wave
        df['qscat'] = flipped_qscat
        df['qext'] = flipped_qext
        df['cos_qscat'] = flipped_cos_qscat

    wave = df['wave'].values.reshape((nradii,nwave)).T
    qscat = df['qscat'].values.reshape((nradii,nwave)).T
    qext = df['qext'].values.reshape((nradii,nwave)).T
    cos_qscat = df['cos_qscat'].values.reshape((nradii,nwave)).T

    # if scattering code returns Q_sca <0 (can happen for extreme examples, like very large particles with very low fractal dimensions), set Q_sca = 1e-16 (basically zero, 
    # but slighty positive so that calc_optics still records opacity in the optical depth array (opd) for purely absorbing cases -- the statement here 
    # checks for opd_sca>0, so it's important to keep it above 0

    qscat[qscat < 0] = 1e-16 # search for any values that are negative, and set them equal to to a very small positive number (1e-16)

    return qext,qscat, cos_qscat, nwave, radii,wave

def get_refrind(igas,directory,aggregates=False): 
    """
    Reads reference files with wavelength, and refractory indicies. The file formats 
    are different for VIRGA and OPTOOL. OPTOOL formats should be created by the user
    and stored in /lnk_data using the "convert_refrind_to_lnk function before running
    "calc_mie_db" function

    VIRGA FILE FORMAT:
    Input files are structured as a 4 column file with columns: 
    index, wavelength (micron), nn, kk 

    OPTOOL FILE FORMAT:
    Input files are structured as a 3 column file with columns: 
    wavelength (micron), nn, kk 

    Parameters
    ----------
    igas : str 
        Gas name 
    directory : str 
        Directory were reference files are located. 

    Returns
    -------
    wavelength, real part, imaginary part 
    """
    if aggregates==False:  # use the VIRGA refractive index database file structure
        filename = os.path.join(directory ,igas+".refrind")
         #put skiprows=1 in loadtxt to skip first line
        try: 
            idummy, wave_in, nn, kk = np.loadtxt(open(filename,'rt').readlines(), unpack=True, usecols=[0,1,2,3])#[:-1]
        except: 
            wave_in, nn, kk = np.loadtxt(open(filename,'rt').readlines(), unpack=True, usecols=[0,1,2], delimiter=',', skiprows=1)
        # if refractive index list is given in ascending order, flip it upside down so that it is descending here (so that it is consistent with the rest of VIRGA)
        if (wave_in[0] < wave_in[-1]): # if first element is smaller than the last one (then it is in ascending order)
            wave_in = np.flipud(wave_in)
            nn = np.flipud(nn) # flip all three arrays to descending order
            kk = np.flipud(kk)

        return wave_in,nn,kk
    
    else: # use the OPTOOL refractive index database file structure
        filename = os.path.join(directory+"/lnk_data",igas+"_VIRGA.lnk") # changed by SEM to .lnk extension for optool use
         #put skiprows=1 in loadtxt to skip first line
        wave_in, nn, kk = np.loadtxt(open(filename,'rt').readlines(), skiprows=1, unpack=True, usecols=[0,1,2])#[:-1]
        return wave_in,nn,kk
    
def get_r_grid(r_min=1e-8, r_max=5.4239131e-2, n_radii=60, log_space=True):
    """
    New version of grid generator - creates lin-spaced or log-spaced radii grids. Updated by MGL 07/01/25.
    
    ALGORITHM DESCRIPTION:

    First, VIRGA makes grid of particle sizes and calculates the optical properties for them. Then, when we find the average particle
    size in a particular layer of cloud, it creates a lognormal distribution of particles (one that has the calculated mean radius),
    and then finds how many particles fall into each of the 'bins' of the radius 'grid' that we have created. Finally, it weights the
    contribution from each bin by the number of particles to calculate the total opacity of that layer. Therefore, the same grid needs
    to be used for making the .mieff files (optical properties) and running VIRGA.

    This new version simplifies the arrays to represent the mean, minimum and maximum values of radius in each bin. It also uses a consistent
    function to calculate the radius values and bin widths, as well as correcting an error for the first bin mean.
    
    Parameters
    ----------

    r_min : float
        minimum radius in grid (in cm). Default is 10^-8 cm (0.0001 um) to match databases in v0.0.
    r_max : float
        maximum radius in grid (in cm). Default is 5.4239131 x 10^-2 cm (542 um), to match databases in v0.0.
    n_radii : int 
        number of increments (note that each of these increments will be further divided into 6 sub-bins to smooth out any resonance features that occur due to specific particle sizes). Default is 60 to match databases in v0.0.
    logspace : boolean
        Default = True. Spaces the radii logarithmically if True (tends to be the case in the clouds). Spaces them linearly if False.
        
    Returns
    -------

    radius : array
        the mean radii for each set of 6 sub-bins
    bin_min : array
        minimum value of each bin
    bin_max :
       maximum value of each bin
    dr : array
        the difference between the start and end of each bin (width of the radius bin)

    """

    # calculate mean radii of bins
    if log_space==True:
        radius = np.logspace(start=np.log10(r_min), stop=np.log10(r_max), num=n_radii) # log-spaced radii is the default. The arguments for np.logspace are the exponents (in base-10) of our r_min and r_max.
    else:
        radius = np.linspace(r_min, r_max, n_radii) # linearly-spaced option (not recommended)

    # calculate dr values
    # this formula can be derived from the following conditions:
    #    1) Bins are linearly centered about the mean radii calculated above
    #    2) Bin widths (dr) are logarithmically spaced, such that the ratio dr[1]/dr[0] = dr[2]/dr[1] = dr[n]/dr[n-1] = constant
    dr = np.zeros(n_radii)
    for i in range(1,n_radii-1):
        dr[i] = 2*(radius[i+1]-radius[i])*(radius[i]-radius[i-1])/(radius[i+1]-radius[i-1]) # bins 1 -> n-1
    dr[0]= 2*(radius[1]-radius[0])-dr[1] # bin 0
    dr[-1]= 2*(radius[-1]-radius[-2])-dr[-2] # bin n

    # calculate the minimum for each radii bin
    bin_min = np.zeros(n_radii)
    for i in range(n_radii):
        bin_min[i] = radius[i] - dr[i]/2 # bin min = mean radius - bin width/2

    # calculate the maximum for each radii bin
    bin_max = np.zeros(n_radii)
    for i in range(n_radii):
        bin_max[i] = radius[i] + dr[i]/2 # bin max = mean radius + bin width/2
    
    return radius, bin_min, bin_max, dr

def get_r_grid_w_max(r_min=1e-8, r_max=5.4239131e-2, n_radii=60):
    """
    Discontinued function. See 'get_r_grid()'.

    ------------------------------------------
    Warning: This function will not work "out of the box" as a substitute for get_r_grid(), because it calculated the bins
    in a different way (rups represents bin boundaries, and the first bin is half the size of the others. This is one
    of the reasons it was discontinued.)
    ------------------------------------------

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

def get_r_grid_legacy(r_min=1e-8, n_radii=60):
    """
    Original code from A&M code. 
    Discontinued function. See 'get_r_grid()'.

    ------------------------------------------
    Warning: This function will not work "out of the box" as a substitute for get_r_grid(), because it calculated the bins
    in a different way (rups represents bin boundaries, and the first bin is half the size of the others. This is one
    of the reasons it was discontinued.)
    ------------------------------------------

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


def picaso_format(opd, w0, g0, pressure=None, wavenumber=None ):
    """
    Gets virga output to picaso format 

    Parameters
    ----------
    opd : ndarray 
        array from virga of extinction optical depths per layer 
    w0 : ndarray 
        array from virga of single scattering albedo 
    g0 : ndarray
        array from virga of asymmetry 
    pressure : array 
        pressure array in bars 
    wavenumber: array 
        wavenumber arry in cm^(-1)

    """
    df = pd.DataFrame(
                dict(opd = opd.flatten(), 
                     w0 = w0.flatten(), 
                     g0 = g0.flatten()))
                     
    if not isinstance(pressure,type(None)):
        df['pressure'] = np.concatenate([[i]*len(wavenumber) for i in pressure])
    if  not isinstance(wavenumber,type(None)):
        df['wavenumber'] = np.concatenate([wavenumber]*len(pressure))
    return df

def picaso_format_custom_wavenumber_grid(opd, w0, g0, wavenumber_grid ):
    """
    This is currently redundant with picaso_format now that picaso_format 
    reads in wavenumber grid. 
    Keeping for now, but will discontinue soon. 
    """
    df = pd.DataFrame(index=[ i for i in range(opd.shape[0]*opd.shape[1])], columns=['pressure','wavenumber','opd','w0','g0'])
    i = 0 
    LVL = []
    WV,OPD,WW0,GG0 =[],[],[],[]
    for j in range(opd.shape[0]):
           for w in range(opd.shape[1]):
                LVL+=[j+1]
                WV+=[wavenumber_grid[w]]
                OPD+=[opd[j,w]]
                WW0+=[w0[j,w]]
                GG0+=[g0[j,w]]
    df.iloc[:,0 ] = LVL
    df.iloc[:,1 ] = WV
    df.iloc[:,2 ] = OPD
    df.iloc[:,3 ] = WW0
    df.iloc[:,4 ] = GG0
    return df

def picaso_format_slab(p_bottom,  opd, w0, g0, 
    wavenumber_grid, pressure_grid ,p_top=None,p_decay=None):
    """
    Sets up a PICASO-readable dataframe that inserts a wavelength dependent aerosol layer at the user's 
    given pressure bounds, i.e., a wavelength-dependent slab of clouds or haze.
    
    Parameters
    ----------
    p_bottom : float 
        the cloud/haze base pressure
        the upper bound of pressure (i.e., lower altitude bound) to set the aerosol layer. (Bars)
    opd : ndarray
        wavelength-dependent optical depth of the aerosol
    w0 : ndarray
        wavelength-dependent single scattering albedo of the aerosol
    g0 : ndarray
        asymmetry parameter = Q_scat wtd avg of <cos theta>
    wavenumber_grid : ndarray
        wavenumber grid in (cm^-1) 
    pressure_grid : ndarray
        bars, user-defined pressure grid for the model atmosphere
    p_top : float
         bars, the cloud/haze-top pressure
         This cuts off the upper cloud region as a step function. 
         You must specify either p_top or p_decay. 
    p_decay : ndarray
        noramlized to 1, unitless
        array the same size as pressure_grid which specifies a 
        height dependent optical depth. The usual format of p_decay is 
        a fsed like exponential decay ~np.exp(-fsed*z/H)


    Returns
    -------
    Dataframe of aerosol layer with pressure (in levels - non-physical units!), wavenumber, opd, w0, and g0 to be read by PICASO
    """
    if (isinstance(p_top, type(None)) & isinstance(p_decay, type(None))): 
        raise Exception("Must specify cloud top pressure via p_top, or the vertical pressure decay via p_decay")
    elif (isinstance(p_top, type(None)) & (~isinstance(p_decay, type(None)))): 
        p_top = 1e-10#arbitarily small pressure to make sure float comparison doest break


    df = pd.DataFrame(index=[ i for i in range(pressure_grid.shape[0]*opd.shape[0])], columns=['pressure','wavenumber','opd','w0','g0'])
    i = 0 
    LVL = []
    WV,OPD,WW0,GG0 =[],[],[],[]
    
    # this loops the opd, w0, and g0 between p and dp bounds and put zeroes for them everywhere else
    for j in range(pressure_grid.shape[0]):
           for w in range(opd.shape[0]):
                #stick in pressure bounds for the aerosol layer:
                if p_top <= pressure_grid[j] <= p_bottom:
                    LVL+=[pressure_grid[j]]
                    WV+=[wavenumber_grid[w]]
                    if isinstance(p_decay,type(None)):
                        OPD+=[opd[w]]
                    else: 
                        OPD+=[p_decay[j]/np.max(p_decay)*opd[w]]
                    WW0+=[w0[w]]
                    GG0+=[g0[w]]
                else:
                    LVL+=[pressure_grid[j]]
                    WV+=[wavenumber_grid[w]]
                    OPD+=[opd[w]*0]
                    WW0+=[w0[w]*0]
                    GG0+=[g0[w]*0]       
                    
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

def recommend_gas(pressure, temperature, mh, mmw, plot=False, returnplot = False, legend='inside', **plot_kwargs):
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
        plot_kwargs['height'] = plot_kwargs.get('plot_height',plot_kwargs.get('height',400))
        plot_kwargs['width'] = plot_kwargs.get('plot_width', plot_kwargs.get('width',600))
        if 'plot_width' in plot_kwargs.keys() : plot_kwargs.pop('plot_width')
        if 'plot_height' in plot_kwargs.keys() : plot_kwargs.pop('plot_height')
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
        if legend == 'inside':
            fig.line(temperature,pressure, legend_label='User',color='black',line_width=5,line_dash='dashed')
            for i in range(ngas):

                fig.line(cond_ts[i],cond_p, legend_label=all_gases[i],color=cols[i],line_width=line_widths[i])
        else:
            f = fig.line(temperature,pressure, color='black',line_width=5,line_dash='dashed')
            legend_it.append(('input profile', [f]))
            for i in range(ngas):

                f = fig.line(cond_ts[i],cond_p ,color=cols[i],line_width=line_widths[i])
                legend_it.append((all_gases[i], [f]))

        if legend == 'outside':
            legend = Legend(items=legend_it, location=(0, 0))
            legend.click_policy="mute"
            fig.add_layout(legend, 'right')   
            
        plot_format(fig)
        
        if returnplot:
            return recommend, fig
        else:
            show(fig)
    
    return recommend

    

def condensation_t(gas_name, mh, mmw, pressure =  np.logspace(-6, 2, 20), gas_mmr=None):
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
                        args=(p, mh, mmw, gas_name, gas_mmr))
        temps += [temp.root]
    return np.array(pressure), np.array(temps)

def hot_jupiter():
    directory = os.path.join(os.path.dirname(__file__), "reference",
                                   "hj.pt")

    df = pd.read_csv(directory,sep=r'\s+', usecols=[1,2,3],
                  names = ['pressure','temperature','kz'],skiprows=1)
    df.loc[df['pressure']>12.8,'temperature'] = np.linspace(1822,2100,df.loc[df['pressure']>12.8].shape[0])
    return df

def brown_dwarf(): 
    directory = os.path.join(os.path.dirname(__file__), "reference",
                                   "t1000g100nc_m0.0.dat")

    df  = pd.read_csv(directory,skiprows=1,sep=r'\s+',
                 header=None, usecols=[1,2,3],
                 names=['pressure','temperature','chf'])
    return df

def warm_neptune(): 
    directory = os.path.join(os.path.dirname(__file__), "reference",
                                   "wn.pt")
def temperate_neptune(): 
    directory = os.path.join(os.path.dirname(__file__), "reference",
                                   "temperate_neptune.pt")

    df  = pd.read_csv(directory,skiprows=0,sep=r'\s+',
                 header=None,
                 names=['pressure','temperature','kz'])
    return df

def convert_refrind_to_lnk(aggregate_list, virga_dir, optool_dir):
    '''
    Converts VIRGA's .refrind files into OPTOOL .lnk files. The differences in file format are below:

        VIRGA: [index,  wavelength,  n,  k]     No header, and in order of descending wavelength

        OPTOOL: [wavelength,  n,  k]     Header (num wavelengths and density, can also include comments), and in order of ascending wavelength

    This function converts the XX.refrind file and saves it in the OPTOOL folder as XX_VIRGA.lnk in the optool directory, ready for optool to 
    use in MMF calculations (see "calc_mie_db").

    Parameters
    ----------
    aggregate_list : list, str
        List of names of gasses. Or a single gas name. 
        See pyeddy.available() to see which ones are currently available. 
    virga_dir: str 
        Directory where you store the VIRGA refractive index files (.refind format).
    optool_dir: str 
        Directory where you store optool refractive index files (.lnk format).
    
    Returns
    -------
    None

    '''

    from virga import gas_properties

    for i in range(len(aggregate_list)):

        aggregate_species = aggregate_list[i]

        # retrieve the density of this species
        run_gas = getattr(gas_properties, aggregate_species) # find density from the gas_properties.py database
        gas_mw, gas_mmr, density = run_gas(1,1,1) # run this function with throwaway values just to obtain the density. Ignore the molecular weight and mass mixing ratio obtained here
        
        # retrieve the refractive indices for this species from the virga database
        refrind_data=pd.read_csv(f'{virga_dir}/{aggregate_species}.refrind', names=['index', 'wavelength', 'n', 'k'], header=None, sep=r'\s+') 
        refrind_data = refrind_data.drop('index', axis=1) # delete the first column (index)
        refrind_data = refrind_data.sort_values('wavelength') # re-order data so that they are in order of ascending wavelength

        with open(f'{optool_dir}/lnk_data/{aggregate_species}_VIRGA.lnk', 'w') as file: # save the new file in the optool directory under the same filename but as a .lnk file
            file.write(f' {len(refrind_data)} {density}\n') # write number of wavelengths and material density at top of file
            refrind_data.to_string(file, col_space=10, index=False, header=None) # print the rest of the data underneath
