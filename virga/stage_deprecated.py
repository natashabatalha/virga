""" Functions that will be deprecated in the next Virga version """

import os

import pandas as pd
import numpy as np


def sdep_vfall_legacy(r, grav,mw_atmos,mfp,visc,t,p, rhop):
    """
    Calculate fallspeed for a spherical particle at one layer in an
    atmosphere, depending on Reynolds number for Stokes flow.

    For Re_Stokes < 1, use Stokes velocity with slip correction
    For Re_Stokes > 1, use fit to Re = exp( b1*x + b2*x^2 )
     where x = log( Cd Re^2 / 24 )
     where b2 = -0.1 (curvature term) and
     b1 from fit between Stokes at Re=1, Cd=24 and Re=1e3, Cd=0.45

    and Precipitation, Reidel, Holland, 1978) and Carlson, Rossow, and
    Orton (J. Atmos. Sci. 45, p. 2066, 1988)

    all units are cgs

    A. Ackerman Feb-2000

    Parameters
    ----------
    r : float
        particle radius (cm)
    grav : float
        acceleration of gravity (cm/s^2)
    mw_atmos : float
        atmospheric molecular weight (g/mol)
    mfp : float
        atmospheric molecular mean free path (cm)
    visc : float
        atmospheric dynamic viscosity (dyne s/cm^2) see Eqn. B2 in A&M
    t : float
        atmospheric temperature (K)
    p  : float
        atmospheric pressure (dyne/cm^2)
    rhop : float
        density of particle (g/cm^3)
    """

    # the drag coefficient for a reynolds number of 1000
    # which is appropriate for oblate spheroids
    # Fig. 10-36 in Pruppacher & Klett 1978
    cdrag = 0.45

    #In order to solve the drag problem we fit y=log(reynolds)
    #as a function of x=log(cdrag * reynolds**2)
    #if you assume that at reynolds= 1, cdrag=24 and
    #reynolds=1000, cdrag=0.45 you get the following fit:
    # y = 0.8 * x - 0.1 * x**2
    #Full explanation: see A & M Appendix B between eq. B2 and B3
    #Simply though, this allows us to get terminal fall velocity from
    #reynolds number
    b1 = 0.8
    b2 = -0.01



    R_GAS = 8.3143e7

    #calculate constants need to get Knudsen and Reynolds numbers
    knudsen = mfp / r
    rho_atmos = p / ( (R_GAS/mw_atmos) * t )
    drho = rhop - rho_atmos

    # Cunningham correction (slip factor for gas kinetic effects)
    # Cunningham, E., "On the velocity of steady fall of spherical particles through fluid
    # medium," Proc. Roy. Soc. A 83(1910)357
    # Cunningham derived a value of 1.26 in the stone ages. In reality, this number is
    # a function of the knudsen number. Various studies have derived
    # different value for this number (see this citation
    # https://www.researchgate.net/publication/242470948_A_Novel_Slip_Correction_Factor_for_Spherical_Aerosol_Particles
    # Within the range of studied values, this 1.26 number changes particle sizes by a few
    # microns
    # That is A OKAY for the level of accuracy we need.
    beta_slip = 1. + 1.26*knudsen

    #Stokes terminal velocity (low Reynolds number)
    #EQN B1 in A&M
    #visc is eqn. B2 in A&M but is computed in `calc_qc`
    #also eqn 10-104 in Pruppacher & klett 1978
    vfall_r = beta_slip*(2.0/9.0)*drho*grav*r**2 / visc

    #compute reynolds number for low reynolds number case
    reynolds = 2.0*r*rho_atmos*vfall_r / visc

    #if reynolds number is between 1-1000 we are in turbulent flow
    #limit
    if 1 < reynolds <= 1e3:
        #OLD METHODLOGY
        #correct drag coefficient for turbulence (x = Cd Re^2 / 24)
        #x = np.log( reynolds )
        #y = b1*x + b2*x**2

        #compute cd * N_re^2 by equating drag and gravitational force
        cd_nre2 = 32.0 * r**3.0 * drho * rho_atmos * grav / (3.0 * visc ** 2 )
        #coefficients from EQN 10-111 in Pruppachar & Klett 1978
        #they are an empirical fit to Figure 10-9
        xx = np.log(cd_nre2)
        b0,b1,b2,b3,b4,b5,b6 = (-0.318657e1, 0.992696, -.153193e-2, -.987059e-3, -.578878e-3,
                                0.855176e-4, -0.327815e-5)
        y = b0 + b1*xx**1 + b2*xx**2 + b3*xx**3 + b4*xx**4 + b5*xx**5 + b6*xx**6

        reynolds = np.exp(y)
        vfall_r = visc*reynolds / (2.*r*rho_atmos)

    if reynolds >1e3 :# 300
        #when Reynolds is greater than 1000, we can just use
        #an asymptotic value that is independent of Reynolds number
        #Eqn. B3 from A&M 01
        vfall_r = beta_slip*np.sqrt( 8.*drho*r*grav / (3.*cdrag*rho_atmos) )

    return vfall_r

def sdep_init_optics(condensibles, nrad=40, rmin=1e-10, read_mie=True):
    """
    Setup up a particle size grid and calculate single-particle scattering
    and absorption efficiencies and other parameters to be used by
    `calc_optics()`

    Parameters
    ----------
    do_optics : bool
        (True/False) Calculate optics (T) or use pre computed files (F)
    read_mie : bool
        (True/False) Read in Mie coefficients from pre compute files `gas_name.mieff`
    condensibles : list of str
        Name in str of all condensible gases e.g. ['H2O','CH4']
    nrad : int
        Number of radius grid points
    rmin : float
        Minimum number of radius grid (cm)

    Returns
    -------
    wave : array
        Wavelength bin centers (cm)
    radius : array
        Radius bin centers (cm)
    dr : array
        Widths of radius bins (cm)
    qscat : array
        Scattering efficiency
    qext : array
        Extinction efficiency
    cos_qscat : array
        qscat * acerage <cos (scattering angle)>
    """
    #equations to compute radius bins for particles
    #these used to be a matrix with different min radii
    raise ValueError("The 'init_optics' function is broken and needs repair before using.")

    # vrat = 2.0
    # pw = 1. / 3.
    # f1 = ( 2*vrat / ( 1 + vrat) )**pw
    # f2 = ( 2 / ( 1 + vrat ) )**pw *  (vrat**pw - 1)
    # radius = rmin * vrat**(irad/3.)
    # rup = f1*radius
    # dr = f2*radius
    #
    # # MGL 04/06/25 -- I HAVE NOT EDITED THIS SECTION FOR v1.0 AS THIS CODE DOES NOT SEEM TO
    # #                 BE USED IN VIRGA. HOWEVER IF IT IS DEVELOPED, THE METHOD OF FINDING
    # #                 THE RADIUS BINS ABOVE SHOULD BE CHANGED TO THE UPDATED METHOD IN THE
    # #                 "get_r_grid()" FUNCTION
    #
    # if read_mie:
    #     #Read extinction and scattering coefficients
    #     #for each condensing vapor
    #     wave, qscat, qext, cos_qscat = get_meiff()
    #
    # else:
    #     #Calculate single-scattering efficiencies etc from refractive indices
    #     #for each condensing vapor
    #
    #     #Mie parameters:
    #     #thetd is angle between incident and scattered radiation
    #     #n_thetd is number of thetd values to consider
    #
    #     thetd = 0.0
    #     n_thetd = 1
    #
    #     #read in refractive indices
    #     for gas in condensibles:
    #         micron_wave, nn, kk = get_refrind(gas)
    #
    #         cm_wave = micron_wave*1e-4
    #
    #         wvno = np.pi*2/cm_wave
    #
    #         for irad in range(nrad):
    #
    #             #subdivide radius grid into 6 bin to average
    #             #out oschilations and call Mie code
    #             if i == 0 :
    #                 dr5 = (rup[0] - radius[0])/5
    #                 rr = radius[0]
    #             else:
    #                 dr5 = (rup[irad] - rup[irad-1])/5
    #                 rr = rup[irad-1]
    #
    #             corerad = 0.0
    #             corereal = 1.0
    #             coreimag = 0.0
    #
    #             for isub in range(6):
    #                 #should return something that is wave x radius
    #                 qext, qscat, cos_qscat = mie_calc(
    #                     rr, nn, kk, thetd,
    #                     n_thetd, corerad, corereal, coreimag, wvno
    #                 )

def sdep_get_refrind(condensible, directory='~/Documents/eddysed/input/optics'):
    """
    Read old style refrind files

    Parameters
    ----------
    condensible : str
        Condensible name (e.g. Al2O3)

    Returns
    -------
    micron_wave, nn, kk as ndarrays
    """
    df = pd.read_csv(
        os.path.join(directory,condensible+'.refrind'),
        skiprows=2, header=None,sep=r'\s+',
        names=['i', 'wavelength', 'nn', 'kk']
    )
    micron_wave=df['wavelength'].values
    nn = df['nn'].values
    kk = df['kk'].values
    return micron_wave, nn, kk


def sdep_fort_mie_calc(RO, RFR, RFI, THET, JX, R, RE2, TMAG2, WVNO):
    """
    Given the refractive indices at a certain wavelength this module
    calculates the Mie scattering by a stratified sphere.The basic code used
    was that described in the report: " Subroutines for computing the parameters of
    the electromagnetic radiation scattered by a sphere " J.V. Dave,
    I B M Scientific Center, Palo Alto , California.
    Report NO. 320 - 3236 .. MAY 1968 .

    Parameters
    ----------
    RO : float
        Outer Shell Radius (cm)
    RFR : float
        Real refractive index of shell layer (in the form n= RFR-i*RFI)
    RFI : float
        Imaginary refractive index of shell layer (in the form n= RFR-i*RFI)
    THET : ndarray
        Angle in degrees between the directions of the incident and the scattered radiation.
    JX : integer
        Total number of THET for which calculations are required
    R : float
        Radius of core (cm)`
    RE2 : float
        Real refractive index of core (in the form n= RE2-i*TMAG2)
    TMAG2 : float
        Imaginary refractive index of core (in the form n= RE2-i*TMAG2)

    WVNO : float
        Wave-number corresponding to the wavelength. (cm^-1)

    Returns
    -------
    QEXT: float
        Efficiency factor for extinction,VAN DE HULST,P.14 ' 127
    QSCAT: float
        Efficiency factor for scattering,VAN DE HULST,P.14 ' 127
    CTBQRS: float
        Average(cos(theta))*QSCAT,VAN DE HULST,P.14 ' 127
    ISTATUS: integer
        Convergence indicator, 0 if converged, -1 if otherwise.
    """

    EPSILON_MIE = 1e-7  ## Tolerance for convergence

    nacap, IT = 1000000, 1

    ACAP=np.zeros(shape=(nacap),dtype=complex)
    W=np.zeros(shape=(3,nacap),dtype=complex)
    WFN=np.zeros(shape=2,dtype=complex)
    Z=np.zeros(shape=4,dtype=complex)
    U=np.zeros(shape=8,dtype=complex)
    T= np.zeros(shape=5)
    TA=np.zeros(shape=4)
    TB=np.zeros(shape=2)
    TC=np.zeros(shape=2)
    TD=np.zeros(shape=2)
    TE=np.zeros(shape=2)
    PI=np.zeros(shape=(3,IT))
    TAU=np.zeros(shape=(3,IT))
    CSTHT=np.zeros(shape=IT)
    THETD=np.full((IT),THET)
    SI2THT=np.zeros(shape=IT)
    ELTRMX=np.zeros(shape=(4,IT,2))
    IFLAG = 1
    if  R/RO < 1e-6:
        IFLAG = 2
    if  JX > IT:
        raise ValueError(
            'THE VALUE OF THE ARGUMENT JX IS GREATER THAN IT. PLEASE '
            'READ COMMENTS.'
        )

    RF =  complex(RFR,-RFI)
    RC =  complex( RE2,-TMAG2 )
    X  =  RO * WVNO
    K1 =  RC * WVNO
    K2 =  RF * WVNO
    K3 =  complex( WVNO, 0.0 )
    Z[0] =  K2 * RO
    Z[1] =  K3 * RO
    Z[2] =  K1 * R
    Z[3] =  K2 * R
    X1   =  Z[0].real
    X4   =  Z[3].real
    Y1   =   Z[0].imag
    Y4   = Z[3].imag
    RRF  =  1.0 / RF
    RX   =  1.0 / X
    RRFX =  RRF * RX
    T[0] = np.sqrt( ( X**2 ) * ( RFR**2 + RFI**2 ))
    NMX1 = int( 1.10 * T[0])
    if  NMX1 > nacap-1 :
        istatus=-1
        return 0,0,0,istatus

    NMX2 = int(T[0])
    if  NMX1 <=  150 :
        NMX1 = 150
        NMX2 = 135

    ACAP[NMX1]  =  complex( 0.0,0.0 ) #+1
    if IFLAG != 2:
        for N in range(3):
            W[N,NMX1]  = complex( 0.0,0.0 ) #+1
    for N in range(NMX1):
        NN = NMX1 - N-1 ## removed a plus 1 to make up for python loop
        ACAP[NN] = (NN+2) * RRFX - 1.0 / ( (NN+2) * RRFX + ACAP[NN+1] )
        if  IFLAG != 2 :
            for M in range(3):
                W[ M,NN ] = ((NN+2) / Z[M+1])  -1.0 / (  (NN+2) / Z[M+1]  +  W[ M,NN+1 ]  )
    for J in range(JX):
        if  THETD[J] < 0.0 :
            THETD[J] =  abs( THETD[J] )

        if  THETD[J] < 90.0:
            T[0]     =  ( 3.14159265359 * THETD[J] ) / 180.0
            CSTHT[J] =  np.cos( T[0])
            SI2THT[J] =  1.0 - CSTHT[J]**2
            continue
        if  THETD[J] == 90.0:
            CSTHT[J]  =  0.0
            SI2THT[J] =  1.0
            continue
        if  THETD[J] > 90.0 :
            raise ValueError(
                ' THE VALUE OF THE SCATTERING ANGLE IS GREATER THAN 90.0 '
                'DEGREES. PLEASE READ COMMENTS'
            )

    for J in range(JX):
        PI[0,J]  =  0.0
        PI[1,J]  =  1.0
        TAU[0,J] =  0.0
        TAU[1,J] =  CSTHT[J]
    T[0]   =  np.cos(X)
    T[1]   =  np.sin(X)
    WM1    =  complex( T[0],-T[1] )
    WFN[0] =  complex( T[1], T[0] )
    TA[0]  =  T[1]
    TA[1]  =  T[0]
    WFN[1] =  RX * WFN[0] - WM1
    TA[2]  =  WFN[1].real
    TA[3]  =  WFN[1].imag
    if IFLAG != 2:
        N=1
        SINX1   =  np.sin( X1 )
        SINX4   =  np.sin( X4 )
        COSX1   =  np.cos( X1 )
        COSX4   =  np.cos( X4 )
        EY1     =  np.exp( Y1 )
        E2Y1    =  EY1 * EY1
        EY4     =  np.exp( Y4 )
        EY1MY4  =  np.exp( Y1 - Y4 )
        EY1PY4  =  EY1 * EY4
        EY1MY4  =  np.exp( Y1 - Y4 )
        AA  =  SINX4 * ( EY1PY4 + EY1MY4 )
        BB  =  COSX4 * ( EY1PY4 - EY1MY4 )
        CC  =  SINX1 * ( E2Y1 + 1.0 )
        DD  =  COSX1 * ( E2Y1 - 1.0 )
        DENOM   =  1.0  +  E2Y1 * ( 4.0 * SINX1 * SINX1 - 2.0 + E2Y1 )
        REALP   =  ( AA * CC  +  BB * DD ) / DENOM
        AMAGP   =  ( BB * CC  -  AA * DD ) / DENOM
        DUMMY   =  complex( REALP, AMAGP )
        AA  =  SINX4 * SINX4 - 0.5
        BB  =  COSX4 * SINX4
        P24H24  =  0.5 + complex( AA,BB ) * EY4 * EY4
        AA  =  SINX1 * SINX4  -  COSX1 * COSX4
        BB  =  SINX1 * COSX4  +  COSX1 * SINX4
        CC  =  SINX1 * SINX4  +  COSX1 * COSX4
        DD  = -SINX1 * COSX4  +  COSX1 * SINX4
        P24H21  =  0.5 * complex( AA,BB ) * EY1 * EY4 + 0.5 * complex( CC,DD ) * EY1MY4
        DH4  =  Z[3] / ( 1.0 + complex( 0.0,1.0 ) * Z[3] )  -  1.0 / Z[3]
        DH1  =  Z[0] / ( 1.0 + complex( 0.0,1.0 ) * Z[0] )  -  1.0 / Z[0]
        DH2  =  Z[1] / ( 1.0 + complex( 0.0,1.0 ) * Z[1] )  -  1.0 / Z[1]
        PSTORE  =  ( DH4 + N / Z[3] )  *  ( W[2,N-1] + N / Z[3] )
        P24H24  =  P24H24 / PSTORE
        HSTORE  =  ( DH1 + N / Z[0] )  *  ( W[2,N-1] + N / Z[3] )
        P24H21  =  P24H21 / HSTORE
        PSTORE  =  ( ACAP[N-1] + N / Z[0] )  /  ( W[2,N-1] + N / Z[3] )
        DUMMY   =  DUMMY * PSTORE
        DUMSQ   =  DUMMY * DUMMY
        U[0] =  K3 * ACAP[N-1]  -  K2 * W[0,N-1]
        U[1] =  K3 * ACAP[N-1]  -  K2 * DH2
        U[2] =  K2 * ACAP[N-1]  -  K3 * W[0,N-1]
        U[3] =  K2 * ACAP[N-1]  -  K3 * DH2
        U[4] =  K1 *  W[2,N-1]  -  K2 * W[1,N-1]
        U[5] =  K2 *  W[2,N-1]  -  K1 * W[1,N-1]
        U[6] =  complex( 0.0,-1.0 )  *  ( DUMMY * P24H21 - P24H24 )
        U[7] =  TA[2] / WFN[1]

        FNA  =  (U[7] * ( U[0]*U[4]*U[6]  +  K1*U[0]  -  DUMSQ*K3*U[4] )
                 /( U[1]*U[4]*U[6]  +  K1*U[1]  -  DUMSQ*K3*U[4] ))
        FNB  =  (U[7] * ( U[2]*U[5]*U[6]  +  K2*U[2]  -  DUMSQ*K2*U[5] )
                 /( U[3]*U[5]*U[6]  +  K2*U[3]  -  DUMSQ*K2*U[5] ))
        TB[0]=FNA.real
        TB[1]=FNA.imag
        TC[0]=FNB.real
        TC[1]=FNB.imag
    elif IFLAG == 2:
        TC1  =  ACAP[0] * RRF  +  RX
        TC2  =  ACAP[0] * RF   +  RX
        FNA  =  ( TC1 * TA[2]  -  TA[0] ) / ( TC1 * WFN[1]  -  WFN[0] )
        FNB  =  ( TC2 * TA[2]  -  TA[0] ) / ( TC2 * WFN[1]  -  WFN[0] )
        TB[0]=FNA.real
        TB[1]=FNA.imag
        TC[0]=FNB.real
        TC[1]=FNB.imag

    FNAP = FNA
    FNBP = FNB
    TD[0]=FNAP.real
    TD[1]=FNAP.imag
    TE[0]=FNBP.real
    TE[1]=FNBP.imag
    T[0] = 1.50

    TB[0] = T[0] * TB[0]
    TB[1] = T[0] * TB[1]
    TC[0] = T[0] * TC[0]
    TC[1] = T[0] * TC[1]

    for  J in range(JX):
        ELTRMX[0,J,0] = TB[0] * PI[1,J] + TC[0] * TAU[1,J]
        ELTRMX[1,J,0] = TB[1] * PI[1,J] + TC[1] * TAU[1,J]
        ELTRMX[2,J,0] = TC[0] * PI[1,J] + TB[0] * TAU[1,J]
        ELTRMX[3,J,0] = TC[1] * PI[1,J] + TB[1] * TAU[1,J]
        ELTRMX[0,J,1] = TB[0] * PI[1,J] - TC[0] * TAU[1,J]
        ELTRMX[1,J,1] = TB[1] * PI[1,J] - TC[1] * TAU[1,J]
        ELTRMX[2,J,1] = TC[0] * PI[1,J] - TB[0] * TAU[1,J]
        ELTRMX[3,J,1] = TC[1] * PI[1,J] - TB[1] * TAU[1,J]


    QEXT   = 2.0 * ( TB[0] + TC[0])
    QSCAT  = ( TB[0]**2 + TB[1]**2 + TC[0]**2 + TC[1]**2 ) / 0.75
    CTBRQS = 0.0
    QBSR   = -2.0*(TC[0] - TB[0])
    QBSI   = -2.0*(TC[1] - TB[1])
    RMM    = -1.0
    N = 2
    while N <= NMX2:
        T[0] = 2*N - 1
        T[1] =   N - 1
        T[2] = 2*N + 1

        for  J in range(JX):
            PI[2,J]  = ( T[0] * PI[1,J] * CSTHT[J] - N * PI[0,J] ) / T[1]
            TAU[2,J] = (CSTHT[J] * ( PI[2,J] - PI[0,J] )  - T[0] * SI2THT[J] * PI[1,J]
                        +  TAU[0,J])
        WM1    =  WFN[0]
        WFN[0] =  WFN[1]
        TA[0]  =  WFN[0].real

        TA[1]  =  WFN[0].imag
        TA[3]  =  WFN[1].imag
        WFN[1] =  T[0] * RX * WFN[0]  -  WM1
        TA[2]  =  WFN[1].real


        if IFLAG != 2:
            DH2  =  - N / Z[1]  +  1.0 / ( (N / Z[1]) - DH2 )
            DH4  =  - N / Z[3]  +  1.0 / ( (N / Z[3]) - DH4 )
            DH1  =  - N / Z[0]  +  1.0 / ( (N / Z[0]) - DH1 )
            PSTORE  =  ( DH4 + (N / Z[3] ))  *  ( W[2,N-1] + (N / Z[3] ))
            P24H24  =  P24H24 / PSTORE
            HSTORE  =  ( DH1 + (N / Z[0] ))  *  ( W[2,N-1] + (N / Z[3] ))
            P24H21  =  P24H21 / HSTORE
            PSTORE  =  ( ACAP[N-1] + (N / Z[0] ))  /  ( W[2,N-1] + (N / Z[3] ))
            DUMMY   =  DUMMY * PSTORE
            DUMSQ   =  DUMMY * DUMMY
            U[0] =  K3 * ACAP[N-1]  -  K2 * W[0,N-1]
            U[1] =  K3 * ACAP[N-1]  -  K2 * DH2
            U[2] =  K2 * ACAP[N-1]  -  K3 * W[0,N-1]
            U[3] =  K2 * ACAP[N-1]  -  K3 * DH2
            U[4] =  K1 *  W[2,N-1]  -  K2 * W[1,N-1]
            U[5] =  K2 *  W[2,N-1]  -  K1 * W[1,N-1]
            U[6] =  complex( 0.0,-1.0 )  *  ( DUMMY * P24H21 - P24H24 )
            U[7] =  TA[2] / WFN[1]

            FNA  =  (U[7] * ( U[0]*U[4]*U[6]  +  K1*U[0]  -  DUMSQ*K3*U[4] )
                     /( U[1]*U[4]*U[6]  +  K1*U[1]  -  DUMSQ*K3*U[4] ))
            FNB  =  (U[7] * ( U[2]*U[5]*U[6]  +  K2*U[2]  -  DUMSQ*K2*U[5] )
                     /( U[3]*U[5]*U[6]  +  K2*U[3]  -  DUMSQ*K2*U[5] ))
            TB[0]=FNA.real
            TB[1]=FNA.imag
            TC[0]=FNB.real
            TC[1]=FNB.imag
        TC1  =  ACAP[N-1] * RRF  +  N * RX
        TC2  =  ACAP[N-1] * RF   +  N * RX
        FN1  =  ( TC1 * TA[2]  -  TA[0] ) /  ( TC1 * WFN[1] - WFN[0] )
        FN2  =  ( TC2 * TA[2]  -  TA[0] ) /  ( TC2 * WFN[1] - WFN[0] )
        M    =  int(WVNO * R)
        if  N >= M :
            if IFLAG ==2:
                FNA  =  FN1
                FNB  =  FN2
                TB[0]=FNA.real
                TB[1]=FNA.imag
                TC[0]=FNB.real
                TC[1]=FNB.imag

            if IFLAG != 2:
                if  abs(  ( FN1-FNA ) / FN1  ) < EPSILON_MIE:
                    if abs(  ( FN2-FNB ) / FN2  )  < EPSILON_MIE :
                        IFLAG = 2



        T[4]  =  N
        T[3]  =  T[0] / ( T[4] * T[1] )
        T[1]  =  (  T[1] * ( T[4] + 1.0 )  ) / T[4]

        CTBRQS +=  T[1] * ( TD[0] * TB[0]  +  TD[1] * TB[1] + TE[0] * TC[0]
                            +  TE[1]* TC[1] )+T[3] * ( TD[0] * TE[0]
                                                       +  TD[1] * TE[1] )
        QEXT   +=    T[2] * ( TB[0] + TC[0] )
        T[3]    =  TB[0]**2 + TB[1]**2 + TC[0]**2 + TC[1]**2
        QSCAT  +=  T[2] * T[3]
        RMM     =  -RMM
        QBSR +=  T[2]*RMM*(TC[0] - TB[0])
        QBSI  +=  T[2]*RMM*(TC[1] - TB[1])

        T[1]    =  N * (N+1)
        T[0]    =  T[2] / T[1]
        K=int(N)
        for J in range(JX):
            ELTRMX[0,J,0] += T[0]*(TB[0]*PI[2,J]+TC[0]*TAU[2,J])
            ELTRMX[1,J,0] += T[0]*(TB[1]*PI[2,J]+TC[1]*TAU[2,J])
            ELTRMX[2,J,0] += T[0]*(TC[0]*PI[2,J]+TB[0]*TAU[2,J])
            ELTRMX[3,J,0] += T[0]*(TC[1]*PI[2,J]+TB[1]*TAU[2,J])
            if  K%2 == 0:
                ELTRMX[0,J,1] += T[0]*(-TB[0]*PI[2,J]+TC[0]*TAU[2,J])
                ELTRMX[1,J,1] += T[0]*(-TB[1]*PI[2,J]+TC[1]*TAU[2,J])
                ELTRMX[2,J,1] += T[0]*(-TC[0]*PI[2,J]+TB[0]*TAU[2,J])
                ELTRMX[3,J,1] += T[0]*(-TC[1]*PI[2,J]+TB[1]*TAU[2,J])
            else:
                ELTRMX[0,J,1] += T[0]*(TB[0]*PI[2,J]-TC[0]*TAU[2,J])
                ELTRMX[1,J,1] += T[0]*(TB[1]*PI[2,J]-TC[1]*TAU[2,J])
                ELTRMX[2,J,1] += T[0]*(TC[0]*PI[2,J]-TB[0]*TAU[2,J])
                ELTRMX[3,J,1] += T[0]*(TC[1]*PI[2,J]-TB[1]*TAU[2,J])

        if  T[3] >= EPSILON_MIE:

            N += 1
            for  J in range(JX):
                PI[0,J]   =   PI[1,J]
                PI[1,J]   =   PI[2,J]
                TAU[0,J]  =  TAU[1,J]
                TAU[1,J]  =  TAU[2,J]


            FNAP  =  FNA
            FNBP  =  FNB
            TD[0]=FNAP.real
            TD[1]=FNAP.imag
            TE[0]=FNBP.real
            TE[1]=FNBP.imag

        else:
            break
    if  N >= NMX2 :
        #print(T[3],NMX2)
        istatus=-1
        return 0,0,0,istatus
    for J in range(JX):
        for K in range(2):
            for I in range(4):
                T[I]  =  ELTRMX[I,J,K]

            ELTRMX[1,J,K]  =      T[0]**2  +  T[1]**2
            ELTRMX[0,J,K]  =      T[2]**2  +  T[3]**2
            ELTRMX[2,J,K]  =  T[0] * T[2]  +  T[1] * T[3]
            ELTRMX[3,J,K]  =  T[1] * T[2]  -  T[3] * T[0]

    T[0]    =    2.0 * RX**2
    QEXT    =   QEXT * T[0]
    QSCAT   =  QSCAT * T[0]
    CTBRQS  =  2.0 * CTBRQS * T[0]
    istatus = 0

    return QEXT,QSCAT,CTBRQS,istatus
