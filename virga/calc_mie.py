""" Main functions for the Virga cloud model """
# pylint: disable=R0902,R0912,R0913,R0914,R0915,R0917,R1702
# pylint: disable=C0103,C0302,C0415,C0201,W0707

import subprocess

import numpy as np
import miepython as mie
import pandas as pd
from .root_functions import calculate_k0, find_N_mon_and_r_mon
from .stage_deprecated import sdep_fort_mie_calc

pi=np.pi

def fort_mie_calc(RO, RFR, RFI, THET, JX, R, RE2, TMAG2, WVNO):
    """ Deprecated function """
    print(
        '[WARNING] "fort_mie_calc" will be fully deprecated in v4. if you are an active '
        'user of this function and there is a rationale we are not aware of for keeping '
        'it please email the developers."'
    )
    return sdep_fort_mie_calc(RO, RFR, RFI, THET, JX, R, RE2, TMAG2, WVNO)

def calc_new_mieff(wave_in, nn,kk, radius, bin_min, bin_max, fort_calc_mie = False):
    #print('binning is back')
    ## Calculates optics by reading refrind files
    thetd=0.0   # incident wave angle
    n_thetd = 1
    #number of radii sub bins in order to smooth out fringe effects
    sub_radii = 6

    # default values
    qe, qs, c_qs = None, None, None

    nradii = len(radius)
    nwave=len(wave_in)  #number of wavalength bin centres for calculation

    qext = np.zeros((nwave,nradii))
    qscat = np.zeros((nwave,nradii))
    cos_qscat = np.zeros((nwave,nradii))

    #compute individual parameters for each gas
    for iwave in range(nwave):
        for irad in range(nradii):

            # create the 6 sub bins
            # calculate the spacing in between sub-bins. These will be evenly spread
            # about the mean radius of each bin.
            dr5 = (bin_max[irad]-bin_min[irad])/(sub_radii-1)
            rr= bin_min[irad] # initial particle radius to test (smallest value in bin, in cm)

            corerad = 0.
            corereal = 1.
            coreimag = 0.
        ## averaging over 6 radial bins to avoid fluctuations
            if fort_calc_mie:
                wave=wave_in*1e-4  ## converting to cm
                bad_mie= False
                for _ in range(sub_radii):
                    wvno=2*np.pi/wave[iwave]
                    qe_pass, qs_pass, c_qs_pass,istatus= fort_mie_calc(
                        rr, nn[iwave], kk[iwave], thetd, n_thetd, corerad,
                        corereal, coreimag, wvno
                    )
                    if istatus == 0:
                        qe=qe_pass
                        qs=qs_pass
                        c_qs=c_qs_pass
                    else:
                        if bad_mie is False :
                            bad_mie = True
                            print ('do_optics(): no Mie solution. So previous grid '
                                   'value assigned')
                            # The mie_calc routine fails to converge if the real
                            # refractive index is smaller than 1. This is true the
                            # fortran counterpart as well. So previous step values
                            # are assigned

                    qext[iwave,irad]+= qe
                    qscat[iwave,irad]+= qs
                    cos_qscat[iwave,irad] += c_qs
                    rr+=dr5
            #this is the default.
            #if no fortran crappy code, use PyMieScatt which does a much faster
            #more robust computation of the Mie parameters
            else:
                wave=wave_in*1e3  ## converting to nm
                ## averaging over 6 radial bins to avoid fluctuations
                for _ in range(sub_radii):
                    #arr = qext, qsca, qback, g,
                    m_eff = nn[iwave]-(1j)*kk[iwave]  # miepython uses negative k convention
                    x_fac = 2 * np.pi * rr*1e7 / wave[iwave]  # both wave and rr are in nm here
                    arr = mie.efficiencies_mx(m_eff, x_fac)

                    qext[iwave,irad]+= arr[0]
                    qscat[iwave,irad]+= arr[1]
                    cos_qscat[iwave,irad] += arr[3]*arr[1]
                    rr+=dr5

            ## adding to master arrays
            qext[iwave,irad] = qext[iwave,irad] / sub_radii
            qscat[iwave,irad] = qscat[iwave,irad] / sub_radii
            cos_qscat[iwave,irad] = cos_qscat[iwave,irad]/ sub_radii

    return qext, qscat, cos_qscat

def calc_new_mieff_optool(wave, radius, bin_min, bin_max, gas,optool_dir,
                          aggregates=False, Df=None, N_mon=None, r_mon=None,
                          k0=0):
    """
    Parameters
    ----------
    wave : array
        wavelength grid, in microns
    radius : array
        effective (mean) particle radius, in centimeters
            -- if aggregates = False, this is the spherical particle radius
            -- if aggregates = True, this is the radius of a sphere of equivalent
               volume to the fractal aggregate
    bin_min : array
        minimum radius in each particle bin
    bin_max : array
        maximum radius in each particle bin
    gas : string
        particle species to condense
    optool_dir : string
        directory location of compiled OPTOOL code. Assumes that .lnk files (the
        optool equivalent to .refrind) are stored in
                a subdirectory within this folder.
                -- NOTE: these .lnk files must be in ascending wavelength order
                         to correctly compute Mie coefficients!
                -- However, we will flip the arrays once calculated, so that
                   .mieff files are consistently saved in order of descending wavelength.
    Df : float
        the fractal dimension, if aggregrates = True
    N_mon : int
        the number of monomers. If None, must set r_mon directly.
    r_mon : float
        monomer particle radius (cm), used if aggregates = True. Can set directly or
        calculate from N_mon.
    k0 : float
        the fractal prefactor, either prescribed by user or calculated using Tazaki (2021) Eq 2

    Returns
    -------
    qext: array
        extinction effiencies for all particle radii/wavelengths
    qsca: array
        scattering effiencies for all particle radii/wavelengths
    cos_qscat: array
        average asymmetry parameter x Q_sca for all particle radii/wavelengths
    """
    nwave=len(wave)
    nradii = len(radius)
    sub_radii = 6

    #set up the empty arrays
    qext = np.zeros((nwave,nradii))#sub_radii
    qscat = np.zeros((nwave,nradii))
    cos_qscat = np.zeros((nwave,nradii))

    # determine whether or not we need to calculate N_mon for each radius
    if N_mon is not None:
        N_mon_prescribed = 1 # N_mon is a fixed value, prescribed by the user
        # record the original set value, in case we need to reduce it
        # temporarily for the smallest particles in the grid
        original_N_mon = N_mon
    else:
        N_mon_prescribed = 0 # N_mon needs to be calculated for each new radius
        original_N_mon = 0 # original number of monomers not set

    # determine whether or not we need to calculate k0 for each radius
    if k0>0:
        k_0_prescribed=1 # k0 is prescribed by the user
    else:
        k_0_prescribed=0 # k0 needs to be calculated for each new radius

    # calculate the largest monomer radius required (so we can find the largest
    # size parameter below!)
    if N_mon_prescribed == 1:
        # if we have provided N_mon, calculate r_mon from the largest value in the
        # array(through conservation of volume between an aggregate and a compact sphere)
        r_mon_max = np.max(radius) / np.cbrt(N_mon)
    else: #user prescribed r_mon
        r_mon_max = r_mon

    # determine largest size parameter required
    min_wavelength = pd.read_csv(
        f"{optool_dir}/lnk_data/{gas}_VIRGA.lnk", skiprows=1, header=None, sep=r'\s+',
        usecols=[0]
    ).values.min() # find minimum wavelength from the lnk data file
    # find largest size parameter: remember to convert largest monomer radius from cm to um
    largest_size_parameter_x = 2.0*np.pi*r_mon_max*1e4/min_wavelength

    if aggregates and largest_size_parameter_x > 200:
        print(f'\n\n\nWARNING!!!! The largest radius is {radius[-1]*10000.0:.3e} um, '
              f'leading to a monomer radius of {r_mon_max*10000.0:.3e} um, and the '
              f'smallest wavelength in the lnk data file is {min_wavelength:.3e} um. '
              'This means that the MONOMER size parameter x_mon will be '
              f'{largest_size_parameter_x:.3f}. Optool will struggle with memory to '
              'calculate anything where x_mon > 200, but this is in the geometric '
              'limit, where Q_ext values are constant. We will therefore assume '
              'constant values above x>200, but because x_mon is wavelength-dependent '
              'we have to send all data to optool one wavelength at a time, which '
              'prevents optool using multiple cores and can slow it down signficantly. '
              'To prevent this decrease the max radius or remove some of the lowest '
              'wavelengths from your .refrind files (especially if, for example, you '
              'have UV/visible wavelengths in there from a big experimental dataset '
              'but arent using them)!\n\n')

        print('\n Mean radius (um)  Sub-bin radius (um)   N_mon    Fractal dimension    '
              'Radius of gyration (um)    k0        r_mon (um)')

        for irad in range(nradii):

            radius_print_switch=0 # reset switch to control which statements are printed to user

            print('----------------------------------------------------------------'
                  '------------------------------------------------------')

            # calculate monomer size in this radius bin (needed to check whether we
            # are in the geometric limit for each wavelength)
            N_mon, r_mon = find_N_mon_and_r_mon(
                N_mon_prescribed, radius[irad], original_N_mon, r_mon
            )
            r_mon_micron = r_mon * 1e4 # convert r_mon from cm into um

            # go through each wavelength one at a time
            for iwave in range(nwave):

                # check whether we are in the geometric limit for this particular
                # wavelength/radius (note that this condition is based on MONOMER radius,
                # not the aggregate compact radius, because that's what the matrix size
                # in MMF depends on)
                # radius and wavelengh are both in um here
                size_parameter_monomer = 2*np.pi*r_mon_micron/wave[iwave]

                # if we have reached the geometric limit (this anticipates and prevents
                # memory overflow)
                if size_parameter_monomer > 200:

                    # check whether a previously calculated value exists
                    if irad>0:

                        # if so, use the last calculated values for every wavelength
                        # (and reshape the output arrays in the same format as output
                        # by PyMieScatt)
                        qext[iwave,irad] += qext[iwave,irad-1]
                        # no need to multiply each of these by subradii (e.g. 6 bins)
                        # because this is BEFORE the smoothing step at the end (we are
                        # still in the wavelength-radius loop)
                        qscat[iwave,irad] += qscat[iwave,irad-1]
                        cos_qscat[iwave,irad] += cos_qscat[iwave,irad-1]

                        if iwave == 0: #only print statement for first wavelength
                            print(f'  {10000.0*radius[irad]:10.5f}          '
                                  f'GEOMETRIC         REGIME:         USING '
                                  f'                PREVIOUS         SUBRADII'
                                  f'      VALUES ')
                    else:
                        print("\n\n\n WARNING: There are no previously calculated Q_ext "
                              "values to use, and we are in the geometric limit. Make "
                              "maximum radius smaller or minimum wavelength larger.")
                        input("Press Enter to continue.")

                else: # for all other cases,run the regular code but for individual wavelengths

                    #create the 6 sub bins
                    # calculate the spacing in between sub-bins. These will be evenly
                    # spread about the mean radius of each bin.
                    dr5 = (bin_max[irad]-bin_min[irad])/(sub_radii-1)
                    dr5_micron = dr5*1e4 # sub-bin radius in um

                    rr= bin_min[irad] # start at minimum of bin (radius in cm)
                    # starting radius of bin in um (we will increase this with each
                    # iteration of sub-bin loop)
                    rr_micron = rr * 1e4

                    #print(f'rr_micron: {rr_micron}')
                    #print(f'dr5_micron: {dr5_micron} um')

                    #loop through the radius sub bins
                    for _ in range(sub_radii):

                        # work out optical parameters at each sub-grid particle size
                        if aggregates:

                            # calculate N_mon and r_mon for this particular sub-radius
                            N_mon, r_mon = find_N_mon_and_r_mon(
                                N_mon_prescribed, rr, original_N_mon, r_mon
                            )
                            r_mon_micron = r_mon * 1e4 # convert r_mon from cm into um

                            # create a list of arguments to pass to optool
                            # SPHERES: single monomer case - basically just doing mie
                            # theory here, but via OPTOOL instead of PyMie
                            if N_mon == 1:
                                # only print statement for first wavelength or if we
                                # were in the geometric regime but are now out of it
                                if iwave == 0 or radius_print_switch==1:
                                    print(f'  {10000.0*radius[irad]:10.5f}         '
                                          f'{rr_micron:10.5f}    {N_mon:10.0f}     '
                                          f'      JUST                   DOING     '
                                          f'       SINGLE       SPHERES ')
                                argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a',
                                                 str(rr_micron),'-l',str(wave[iwave]),'-mie',
                                                 '-q'] # Mie theory (single spherical particle)
                            elif rr_micron <= r_mon_micron:
                                N_mon=1 # SPHERES: aggregates cannot be smaller than 1
                                # monomer, so just assume spheres if r < r_mon. Also,
                                # set N_mon = 1 (this is not currently used in any further
                                # calculations, but is just added here to be clear to the
                                # user about what we are doing, as well as being best
                                # practice in case we do add code below this part)

                                # only print statement for first wavelength or if we were
                                # in the geometric regime but are now out of it
                                if iwave == 0 or radius_print_switch == 1:
                                    print(f'  {10000.0*radius[irad]:10.5f}         '
                                          f'{rr_micron:10.5f}    {N_mon:10.0f}     '
                                          f'      JUST                   DOING     '
                                          f'       SINGLE       SPHERES ')
                                argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a',
                                                 str(rr_micron),'-l',str(wave[iwave]),'-mie',
                                                 '-q'] # Mie theory (single spherical particle)

                            else: # use MMF for aggregates

                                # if fractal prefactor k0 is left unprescribed by user,
                                # calculate it here
                                if k_0_prescribed == 0:
                                    k0 = calculate_k0(N_mon, Df)

                                # calculate radius of gyration (in um)
                                r_gyro = (N_mon/k0)**(1/Df) * r_mon_micron

                                # only print statement for first wavelength or if we were
                                # in the geometric regime but are now out of it
                                if iwave == 0 or radius_print_switch==1:
                                    print(f'  {10000.0*radius[irad]:10.5f}         '
                                          f'{rr_micron:10.5f}    {N_mon:10.0f}      '
                                          f'{Df:10.3f}              {r_gyro:10.5f}      '
                                          f'{k0:10.3f}      {r_mon_micron:10.7f} ')
                                # MMF theory (aggregates): -c, link to refractive indices,
                                # -a, radius, -l, wavelength, -mmf, monomer radius r0,
                                # fractal dimension d_f, fractal prefactor k0
                                argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a',
                                                 str(rr_micron),'-l',str(wave[iwave]),'-mmf',
                                                 str(r_mon_micron),str(Df),str(k0), '-q']
                        else:
                            # only print statement for first wavelength or if we were
                            # in the geometric regime but are now out of it
                            if iwave==0 or radius_print_switch==1:
                                print(f'  {10000.0*radius[irad]:10.5f}         '
                                      f'{rr_micron:10.5f}    {N_mon:10.0f}      '
                                      f'     JUST                   DOING        '
                                      f'    SINGLE       SPHERES ')
                            # Mie theory (single spherical particle)
                            argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a',
                                             str(rr_micron),'-l',str(wave[iwave]),
                                             '-mie', '-q']

                        # run optool using the string created above - this finds kext[0],
                        # ksca[0], kabs[0], gsca[0], one particle radius at a time, for all
                        # wavelengths in the list, and stores them in a table within file
                        # "dustkappa.dat" (written in the optool directory)
                        # code executable + arguments, complete in the working directory of
                        # the compiled OPTOOL code
                        subprocess.run(
                            [f"{optool_dir}/optool"]+argument_list, cwd=optool_dir
                        )

                        # read the wavelength, k_abs, k_sca, and g_asymmetry values in the
                        # 'dustkappa.dat' file that was just created by optool
                        # skip the comments in the header of the refractive index file, and
                        # read in the data
                        arr=pd.read_csv(
                            optool_dir+"/dustkappa.dat",
                            names=['lambda','k_abs','k_sca','g_asymmetry'], delimiter=r"\s+",
                            comment='#')
                        # drop first two rows (data not needed) and re-index
                        arr= arr.drop([0,1]).reset_index(drop=True)

                        # read the density from the header of the material file
                        # read the line after the comments and before the data in the
                        # refractive index file (contains "num wavelengths, density"
                        # in that order)
                        density=pd.read_csv(
                            optool_dir+'/lnk_data/'+gas+'_VIRGA.lnk',nrows=1, delimiter=r"\s+",
                            comment='#', header=None)
                        rho_p = density[1][0] # store the density of the condensate in g/cm3

                        #convert k_abs, k_sca and g from optool output into Q values.
                        # Radius needs to be in centimeters here for this conversion
                        # since rho_p has g/cm3 units
                        qe = (4 * (arr["k_abs"]+arr["k_sca"]) * rho_p * rr) / 3
                        qs = (4 * arr["k_sca"] * rho_p * rr) / 3
                        # qa = (4 * arr["k_abs"] * rho_p * rr) / 3
                        cos_qs = arr["g_asymmetry"] * qs

                        #increment over the radius sub bins
                        rr += dr5
                        rr_micron += dr5_micron

                        #reshape the output arrays in the same format as output by
                        # PyMieScatt -- this will sum all values for all 6 sub-bins
                        # into a single array element, and we find te average of the
                        # whole array outside of the radius loop
                        qext[iwave,irad] += float(qe.iloc[0])
                        qscat[iwave,irad] += float(qs.iloc[0])
                        cos_qscat[iwave,irad] += float(cos_qs.iloc[0])

                    # reset witch to control which statements are printed to user
                    radius_print_switch=0

        # smooth over all the radius bins
        # matrix as a function of [iwave,irad] divide by 6 (the number of subradii bins)
        qext = qext / sub_radii
        qscat = qscat / sub_radii  # matrix as a function of [iwave,irad]
        cos_qscat = cos_qscat / sub_radii  # matrix as a function of [iwave,irad]

    else: # regular method (no size parameter mitigation needed to prevent memory overflow)

        print('\n Mean radius (um)  Sub-bin radius (um)   N_mon    Fractal dimension    '
              'Radius of gyration (um)    k0        r_mon (um)')

        for irad in range(nradii):

            print('--------------------------------------------------------------------'
                  '--------------------------------------------------')

            #create the 6 sub bins
            # calculate the spacing in between sub-bins. These will be evenly spread
            # about the mean radius of each bin.
            dr5 = (bin_max[irad]-bin_min[irad])/(sub_radii-1)
            dr5_micron = dr5*1e4 # sub-bin radius in um

            rr= bin_min[irad] # start at minimum of bin (radius in cm)
            rr_micron = rr * 1e4 # starting radius of bin in um (we will increase this with
            # each iteration of sub-bin loop)

            #print(f'rr_micron: {rr_micron}')
            #print(f'dr5_micron: {dr5_micron} um')

            #loop through the radius sub bins
            for _ in range(sub_radii):

                # work out optical parameters at each sub-grid particle size
                if aggregates:

                    # calculate N_mon and r_mon for this particular sub-radius
                    N_mon, r_mon = find_N_mon_and_r_mon(
                        N_mon_prescribed, rr, original_N_mon, r_mon
                    )
                    r_mon_micron = r_mon * 1e4 # convert r_mon from cm into um

                    # create a list of arguments to pass to optool
                    if N_mon == 1: # SPHERES: single monomer case - basically just doing mie
                        # theory here, but via OPTOOL instead of PyMie
                        print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    '
                              f'{N_mon:10.0f}           JUST                   DOING        '
                              f'    SINGLE       SPHERES ')
                        argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a',
                                         str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk',
                                         '-mie', '-q'] # Mie theory (single spherical particle)
                    elif rr_micron <= r_mon_micron:
                        N_mon=1 # SPHERES: aggregates cannot be smaller than 1 monomer, so
                        # just assume spheres if r < r_mon. Also, set N_mon = 1 (this is
                        # not currently used in any further calculations, but is just added
                        # here to be clear to the user about what we are doing, as well as
                        # being best practice in case we do add code below this part)
                        print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    '
                              f'{N_mon:10.0f}           JUST                   DOING        '
                              f'    SINGLE       SPHERES ')
                        argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk',
                                         '-a', str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk',
                                         '-mie', '-q'] # Mie theory (single spherical particle)

                    else: # use MMF for aggregates

                        # if fractal prefactor k0 is left unprescribed by user, calculate
                        # it here
                        if k_0_prescribed == 0:
                            k0 = calculate_k0(N_mon, Df)

                        # calculate radius of gyration (in um)
                        r_gyro = (N_mon/k0)**(1/Df) * r_mon_micron

                        print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    '
                              f'{N_mon:10.0f}      {Df:10.3f}              {r_gyro:10.5f}      '
                              f'{k0:10.3f}      {r_mon_micron:10.7f} ')
                        # MMF theory (aggregates): -c, link to refractive indices, -a,
                        # radius, -l, list of wavelengths (do whole file), -mmf, monomer
                        # radius r0, fractal dimension d_f, fractal prefactor k0
                        argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk',
                                         '-a', str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk',
                                         '-mmf',str(r_mon_micron),str(Df),str(k0), '-q']
                else:
                    # Mie theory (single spherical particle)
                    print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    '
                          f'{N_mon:10.0f}           JUST                   DOING        '
                          f'    SINGLE       SPHERES ')
                    argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a',
                                     str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk',
                                     '-mie', '-q']

                # run optool using the string created above - this finds kext[0], ksca[0],
                # kabs[0], gsca[0], one particle radius at a time, for all wavelengths in
                # the list, and stores them in a table within file "dustkappa.dat" (written
                # in the optool directory)
                # code executable + arguments, complete in the working directory of the
                # compiled OPTOOL code
                subprocess.run([f"{optool_dir}/optool"]+argument_list, cwd=optool_dir)

                # read the wavelengths, k_abs, k_sca, and g_asymmetry values in the
                # 'dustkappa.dat' file that was just created by optool
                # skip the comments in the header of the refractive index file, and read in
                # the data
                arr=pd.read_csv(
                    optool_dir+"/dustkappa.dat", names=['lambda','k_abs','k_sca','g_asymmetry'],
                    delimiter=r"\s+", comment='#')
                # drop first two rows (data not needed) and re-index
                arr= arr.drop([0,1]).reset_index(drop=True)

                #print(arr)

                # read the density from the header of the material file
                # read the line after the comments and before the data in the refractive
                # index file (contains "num wavelengths, density" in that order)
                density=pd.read_csv(
                    optool_dir+'/lnk_data/'+gas+'_VIRGA.lnk',nrows=1, delimiter=r"\s+",
                    comment='#', header=None)
                rho_p = density[1][0] # store the density of the condensate in g/cm3

                # convert k_abs, k_sca and g from optool output into Q values. Radius needs to
                # be in centimeters here for this conversion since rho_p has g/cm3 units
                qe = (4 * (arr["k_abs"]+arr["k_sca"]) * rho_p * rr) / 3
                qs = (4 * arr["k_sca"] * rho_p * rr) / 3
                # qa = (4 * arr["k_abs"] * rho_p * rr) / 3
                cos_qs = arr["g_asymmetry"] * qs

                #increment over the radius sub bins
                rr += dr5
                rr_micron += dr5_micron

                #reshape the output arrays in the same format as output by PyMieScatt
                for iwave in range(nwave):
                    qext[iwave,irad] += qe[iwave]
                    qscat[iwave,irad] += qs[iwave]
                    cos_qscat[iwave,irad] += cos_qs[iwave]

        # smooth over all the radius bins
        # matrix as a function of [iwave,irad] divide by 6 (the number
        # of subradii bins)
        qext = qext / sub_radii
        qscat = qscat / sub_radii  #matrix as a function of [iwave,irad]
        cos_qscat = cos_qscat / sub_radii  #matrix as a function of [iwave,irad]

    return qext, qscat, cos_qscat
