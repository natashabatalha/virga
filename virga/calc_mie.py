import numpy as np
pi=np.pi
import miepython as mie
import os 
import subprocess
import pandas as pd
from .root_functions import calculate_k0, find_N_mon_and_r_mon

def fort_mie_calc(RO, RFR, RFI, THET, JX, R, RE2, TMAG2, WVNO ):
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
    WFN=np.zeros(shape=(2),dtype=complex)
    Z=np.zeros(shape=(4),dtype=complex)
    U=np.zeros(shape=(8),dtype=complex)
    T= np.zeros(shape=(5))    
    TA=np.zeros(shape=(4))     
    TB=np.zeros(shape=(2))        
    TC=np.zeros(shape=(2))
    TD=np.zeros(shape=(2))  
    TE=np.zeros(shape=(2))  
    PI=np.zeros(shape=(3,IT))   
    TAU=np.zeros(shape=(3,IT))
    CSTHT=np.zeros(shape=(IT)) 
    THETD=np.full((IT),THET) 
    SI2THT=np.zeros(shape=(IT))   
    ELTRMX=np.zeros(shape=(4,IT,2))
    IFLAG = 1
    if  R/RO < 1e-6:    
        IFLAG = 2
    if  JX > IT:
        raise Exception('THE VALUE OF THE ARGUMENT JX IS GREATER THAN IT. PLEASE READ COMMENTS.')
        
            
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
            raise Exception(' THE VALUE OF THE SCATTERING ANGLE IS GREATER THAN 90.0 DEGREES. PLEASE READ COMMENTS')
        
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
    TA[2]  =  (WFN[1].real)
    TA[3]  =  (WFN[1].imag)
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
            
        FNA  =  U[7] * ( U[0]*U[4]*U[6]  +  K1*U[0]  -  DUMSQ*K3*U[4] ) /( U[1]*U[4]*U[6]  +  K1*U[1]  -  DUMSQ*K3*U[4] )
        FNB  =  U[7] * ( U[2]*U[5]*U[6]  +  K2*U[2]  -  DUMSQ*K2*U[5] ) /( U[3]*U[5]*U[6]  +  K2*U[3]  -  DUMSQ*K2*U[5] )
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
            TAU[2,J] = CSTHT[J] * ( PI[2,J] - PI[0,J] )  - T[0] * SI2THT[J] * PI[1,J]  +  TAU[0,J]
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

            FNA  =  U[7] * ( U[0]*U[4]*U[6]  +  K1*U[0]  -  DUMSQ*K3*U[4] ) /( U[1]*U[4]*U[6]  +  K1*U[1]  -  DUMSQ*K3*U[4] )
            FNB  =  U[7] * ( U[2]*U[5]*U[6]  +  K2*U[2]  -  DUMSQ*K2*U[5] ) /( U[3]*U[5]*U[6]  +  K2*U[3]  -  DUMSQ*K2*U[5] )
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

        CTBRQS +=  T[1] * ( TD[0] * TB[0]  +  TD[1] * TB[1] + TE[0] * TC[0]  +  TE[1]* TC[1] )+T[3] * ( TD[0] * TE[0]  +  TD[1] * TE[1] )
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


def calc_new_mieff(wave_in, nn,kk, radius, bin_min, bin_max, fort_calc_mie = False):
    #print('binning is back')
    ## Calculates optics by reading refrind files
    thetd=0.0   # incident wave angle
    n_thetd=1
    #number of radii sub bins in order to smooth out fringe effects 
    sub_radii = 6
    
    nradii = len(radius)
    nwave=len(wave_in)  #number of wavalength bin centres for calculation
    
    qext = np.zeros((nwave,nradii))
    qscat = np.zeros((nwave,nradii))
    cos_qscat = np.zeros((nwave,nradii))
        
    #compute individual parameters for each gas
    for iwave in range(nwave):
        for irad in range(nradii):

            #create the 6 sub bins 
            dr5 = (bin_max[irad]-bin_min[irad])/(sub_radii-1) # calculate the spacing in between sub-bins. These will be evenly spread about the mean radius of each bin.        
            rr= bin_min[irad] # initial particle radius to test (smallest value in bin, in cm)

            corerad = 0.
            corereal = 1.
            coreimag = 0.
        ## averaging over 6 radial bins to avoid fluctuations
            if fort_calc_mie:
                wave=wave_in*1e-4  ## converting to cm 
                bad_mie= False
                for isub in range(sub_radii):
                            wvno=2*np.pi/wave[iwave]
                            qe_pass, qs_pass, c_qs_pass,istatus= fort_mie_calc(rr, nn[iwave], kk[iwave], thetd, n_thetd, corerad, corereal, coreimag, wvno)
                            if istatus == 0:
                                    qe=qe_pass
                                    qs=qs_pass
                                    c_qs=c_qs_pass
                            else:
                                if bad_mie == False :
                                    bad_mie = True
                                    print ('do_optics(): no Mie solution. So previous grid value assigned')
                                    ## The mie_calc routine fails to converge if the real refractive index is smaller than 1. This is true the
                                    ## fortran counterpart as well. So previous step values are assigned

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
                for isub in range(sub_radii):
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

def calc_new_mieff_optool(wave,radius,bin_min,bin_max,gas,optool_dir,aggregates=False,Df=None,N_mon=None,r_mon=None,k0=0):
    """
    Parameters
    ----------
    wave : array
        wavelength grid, in microns
    radius : array
        effective (mean) particle radius, in centimeters 
            -- if aggregates = False, this is the spherical particle radius
            -- if aggregates = True, this is the radius of a sphere of equivalent volume to the fractal aggregate
    bin_min : array
        minimum radius in each particle bin
    bin_max : array
        maximum radius in each particle bin
    gas : string
        particle species to condense
    optool_dir : string
        directory location of compiled OPTOOL code. Assumes that .lnk files (the optool equivalent to .refrind) are stored in
                a subdirectory within this folder.
                -- NOTE: these .lnk files must be in ascending wavelength order to correctly compute Mie coefficients! 
                -- However, we will flip the arrays once calculated, so that .mieff files are consistently saved in order of descending wavelength.
    Df : float
        the fractal dimension, if aggregrates = True
    N_mon : int
        the number of monomers. If None, must set r_mon directly.
    r_mon : float
        monomer particle radius (cm), used if aggregates = True. Can set directly or calculate from N_mon.
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
        N_mon_prescribed=1 # N_mon is a fixed value, prescribed by the user
        original_N_mon = N_mon # record the original set value, in case we need to reduce it temporarily for the smallest particles in the grid
    else:
        N_mon_prescribed=0 # N_mon needs to be calculated for each new radius
        original_N_mon = 0 # original number of monomers not set

    # determine whether or not we need to calculate k0 for each radius
    if k0>0:
        k_0_prescribed=1 # k0 is prescribed by the user
    else:
        k_0_prescribed=0 # k0 needs to be calculated for each new radius

    # calculate the largest monomer radius required (so we can find the largest size parameter below!)
    if (N_mon_prescribed==1):
        r_mon_max = np.max(radius) / np.cbrt(N_mon) # if we have provided N_mon, calculate r_mon from the largest value in the array(through conservation of volume between an aggregate and a compact sphere)
    else: #user prescribed r_mon
        r_mon_max = r_mon

    # determine largest size parameter required
    min_wavelength = pd.read_csv(f"{optool_dir}/lnk_data/{gas}_VIRGA.lnk", skiprows=1, header=None, sep='\s+', usecols=[0]).values.min() # find minimum wavelength from the lnk data file
    largest_size_parameter_x = 2.0*np.pi*r_mon_max*1e4/min_wavelength # find largest size parameter: remember to convert largest monomer radius from cm to um
    
    if (aggregates==True) and (largest_size_parameter_x > 200):
        print(f'\n\n\nWARNING!!!! The largest radius is {radius[-1]*10000.0:.3e} um, leading to a monomer radius of {r_mon_max*10000.0:.3e} um, and the smallest wavelength')
        print(f'in the lnk data file is {min_wavelength:.3e} um. This means that the MONOMER size parameter x_mon will be {largest_size_parameter_x:.3f}. Optool will struggle')
        print(f'with memory to calculate anything where x_mon > 200, but this is in the geometric limit, where Q_ext values are constant. We')
        print('will therefore assume constant values above x>200, but because x_mon is wavelength-dependent we have to send all data to optool')
        print('one wavelength at a time, which prevents optool using multiple cores and can slow it down signficantly. To prevent this decrease')
        print(f'the max radius or remove some of the lowestwavelengths from your .refrind files (especially if, for example, you have UV/visible')
        print("wavelengths in there from a big experimental dataset but aren't using them)!\n\n")

        print(f'\n Mean radius (um)  Sub-bin radius (um)   N_mon    Fractal dimension    Radius of gyration (um)    k0        r_mon (um)')



        for irad in range(nradii):

            radius_print_switch=0 # reset switch to control which statements are printed to user

            print('----------------------------------------------------------------------------------------------------------------------')
     
            # calculate monomer size in this radius bin (needed to check whether we are in the geometric limit for each wavelength)
            N_mon, r_mon = find_N_mon_and_r_mon(N_mon_prescribed, radius[irad], original_N_mon, r_mon)
            r_mon_micron = r_mon * 1e4 # convert r_mon from cm into um

            # go through each wavelength one at a time
            for iwave in range(nwave): 

                # check whether we are in the geometric limit for this particular wavelength/radius (note that this condition is based on MONOMER radius, not the aggregate compact radius, because that's what the matrix size in MMF depends on)
                size_parameter_monomer = 2*np.pi*r_mon_micron/wave[iwave] # radius and wavelengh are both in um here
                
                # if we have reached the geometric limit (this anticipates and prevents memory overflow)
                if (size_parameter_monomer>200):
                    
                    # check whether a previously calculated value exists
                    if irad>0:
                            
                        # if so, use the last calculated values for every wavelength (and reshape the output arrays in the same format as output by PyMieScatt)
                        qext[iwave,irad] += qext[iwave,irad-1]
                        qscat[iwave,irad] += qscat[iwave,irad-1] # no need to multiply each of these by subradii (e.g. 6 bins) because this is BEFORE the smoothing step at the end (we are still in the wavelength-radius loop)
                        cos_qscat[iwave,irad] += cos_qscat[iwave,irad-1]
                        
                        if(iwave==0): #only print statement for first wavelength
                            print(f'  {10000.0*radius[irad]:10.5f}          GEOMETRIC         REGIME:         USING                 PREVIOUS         SUBRADII      VALUES ')
                            radius_print_switch=1 # set switch to 1 so that we still print the regular statements for the non-geometric regime radii below
                    else:
                        print("\n\n\n WARNING: There are no previously calculated Q_ext values to use, and we are in the geometric limit. Make maximum radius smaller or minimum wavelength larger.")
                        wait = input("Press Enter to continue.")

                else: # for all other cases,run the regular code but for individual wavelengths

                    #create the 6 sub bins 
                    dr5 = (bin_max[irad]-bin_min[irad])/(sub_radii-1) # calculate the spacing in between sub-bins. These will be evenly spread about the mean radius of each bin.
                    dr5_micron = dr5*1e4 # sub-bin radius in um
                    
                    rr= bin_min[irad] # start at minimum of bin (radius in cm)
                    rr_micron = rr * 1e4 # starting radius of bin in um (we will increase this with each iteration of sub-bin loop)

                    #print(f'rr_micron: {rr_micron}')
                    #print(f'dr5_micron: {dr5_micron} um')

                    #loop through the radius sub bins
                    for isub in range(sub_radii):

                        # work out optical parameters at each sub-grid particle size
                        if (aggregates==True):

                            # calculate N_mon and r_mon for this particular sub-radius
                            N_mon, r_mon = find_N_mon_and_r_mon(N_mon_prescribed, rr, original_N_mon, r_mon)
                            r_mon_micron = r_mon * 1e4 # convert r_mon from cm into um

                            # create a list of arguments to pass to optool
                            if(N_mon==1): # SPHERES: single monomer case - basically just doing mie theory here, but via OPTOOL instead of PyMie
                                if(iwave==0) or (radius_print_switch==1): #only print statement for first wavelength or if we were in the geometric regime but are now out of it
                                    print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}           JUST                   DOING            SINGLE       SPHERES ')
                                argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l',str(wave[iwave]),'-mie', '-q'] # Mie theory (single spherical particle)
                            elif rr_micron <= r_mon_micron: 
                                N_mon=1 # SPHERES: aggregates cannot be smaller than 1 monomer, so just assume spheres if r < r_mon. Also, set N_mon = 1 (this is not currently used in any further calculations, but is just added here to be clear to the user about what we are doing, as well as being best practice in case we do add code below this part)
                                if(iwave==0) or (radius_print_switch==1): #only print statement for first wavelength or if we were in the geometric regime but are now out of it
                                    print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}           JUST                   DOING            SINGLE       SPHERES ')
                                argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l',str(wave[iwave]),'-mie', '-q'] # Mie theory (single spherical particle)

                            else: # use MMF for aggregates

                                # if fractal prefactor k0 is left unprescribed by user, calculate it here
                                if(k_0_prescribed==0):
                                    k0 = calculate_k0(N_mon, Df)
                                
                                # calculate radius of gyration (in um)
                                r_gyro = (N_mon/k0)**(1/Df) * r_mon_micron 

                                if(iwave==0) or (radius_print_switch==1): #only print statement for first wavelength or if we were in the geometric regime but are now out of it
                                    print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}      {Df:10.3f}              {r_gyro:10.5f}      {k0:10.3f}      {r_mon_micron:10.7f} ')
                                argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l',str(wave[iwave]),'-mmf',str(r_mon_micron),str(Df),str(k0), '-q'] # MMF theory (aggregates): -c, link to refractive indices, -a, radius, -l, wavelength, -mmf, monomer radius r0, fractal dimension d_f, fractal prefactor k0
                        else:
                                if(iwave==0) or (radius_print_switch==1): #only print statement for first wavelength or if we were in the geometric regime but are now out of it
                                    print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}           JUST                   DOING            SINGLE       SPHERES ')
                                argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l',str(wave[iwave]),'-mie', '-q'] # Mie theory (single spherical particle)

                        # run optool using the string created above - this finds kext[0], ksca[0], kabs[0], gsca[0], one particle radius at a time, for all wavelengths in the list, and stores them in a table within file "dustkappa.dat" (written in the optool directory)
                        subprocess.run([f"{optool_dir}/optool"]+argument_list, cwd=optool_dir) # code executable + arguments, complete in the working directory of the compiled OPTOOL code 

                        # read the wavelength, k_abs, k_sca, and g_asymmetry values in the 'dustkappa.dat' file that was just created by optool
                        arr=pd.read_csv(optool_dir+"/dustkappa.dat", names=['lambda','k_abs','k_sca','g_asymmetry'], delimiter=r"\s+", comment='#') # skip the comments in the header of the refractive index file, and read in the data
                        arr= arr.drop([0,1]).reset_index(drop=True) # drop first two rows (data not needed) and re-index

                        # read the density from the header of the material file
                        density=pd.read_csv(optool_dir+'/lnk_data/'+gas+'_VIRGA.lnk',nrows=1, delimiter=r"\s+", comment='#', header=None) # read the line after the comments and before the data in the refractive index file (contains "num wavelengths, density" in that order)
                        rho_p = density[1][0] # store the density of the condensate in g/cm3

                        #convert k_abs, k_sca and g from optool output into Q values. Radius needs to be in centimeters here for this conversion since rho_p has g/cm3 units
                        qe = (4 * (arr["k_abs"]+arr["k_sca"]) * rho_p * rr) / 3
                        qs = (4 * arr["k_sca"] * rho_p * rr) / 3
                        qa = (4 * arr["k_abs"] * rho_p * rr) / 3
                        cos_qs = arr["g_asymmetry"] * qs

                        #increment over the radius sub bins
                        rr += dr5  
                        rr_micron += dr5_micron
                        
                        #reshape the output arrays in the same format as output by PyMieScatt -- this will sum all values for all 6 sub-bins into a single array element, and we find te average of the whole array outside of the radius loop
                        qext[iwave,irad] += float(qe.iloc[0])
                        qscat[iwave,irad] += float(qs.iloc[0])
                        cos_qscat[iwave,irad] += float(cos_qs.iloc[0])
                    
                    radius_print_switch=0 # reset witch to control which statements are printed to user
        
        #smooth over all the radius bins 
        qext = qext / sub_radii  #matrix as a function of [iwave,irad] divide by 6 (the number of subradii bins)
        qscat = qscat / sub_radii  #matrix as a function of [iwave,irad]   
        cos_qscat = cos_qscat / sub_radii  #matrix as a function of [iwave,irad]   

    else: # regular method (no size parameter mitigation needed to prevent memory overflow)

        print(f'\n Mean radius (um)  Sub-bin radius (um)   N_mon    Fractal dimension    Radius of gyration (um)    k0        r_mon (um)')

        for irad in range(nradii):
                    
            print('----------------------------------------------------------------------------------------------------------------------')

            #create the 6 sub bins 
            dr5 = (bin_max[irad]-bin_min[irad])/(sub_radii-1) # calculate the spacing in between sub-bins. These will be evenly spread about the mean radius of each bin.
            dr5_micron = dr5*1e4 # sub-bin radius in um
            
            rr= bin_min[irad] # start at minimum of bin (radius in cm)
            rr_micron = rr * 1e4 # starting radius of bin in um (we will increase this with each iteration of sub-bin loop)

            #print(f'rr_micron: {rr_micron}')
            #print(f'dr5_micron: {dr5_micron} um')

            #loop through the radius sub bins
            for isub in range(sub_radii):

                # work out optical parameters at each sub-grid particle size
                if (aggregates==True):

                    # calculate N_mon and r_mon for this particular sub-radius
                    N_mon, r_mon = find_N_mon_and_r_mon(N_mon_prescribed, rr, original_N_mon, r_mon)
                    r_mon_micron = r_mon * 1e4 # convert r_mon from cm into um

                    # create a list of arguments to pass to optool
                    if(N_mon==1): # SPHERES: single monomer case - basically just doing mie theory here, but via OPTOOL instead of PyMie
                        print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}           JUST                   DOING            SINGLE       SPHERES ')
                        argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk','-mie', '-q'] # Mie theory (single spherical particle)
                    elif rr_micron <= r_mon_micron: 
                        N_mon=1 # SPHERES: aggregates cannot be smaller than 1 monomer, so just assume spheres if r < r_mon. Also, set N_mon = 1 (this is not currently used in any further calculations, but is just added here to be clear to the user about what we are doing, as well as being best practice in case we do add code below this part)
                        print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}           JUST                   DOING            SINGLE       SPHERES ')
                        argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk','-mie', '-q'] # Mie theory (single spherical particle)

                    else: # use MMF for aggregates

                        # if fractal prefactor k0 is left unprescribed by user, calculate it here
                        if(k_0_prescribed==0):
                            k0 = calculate_k0(N_mon, Df)
                        
                        # calculate radius of gyration (in um)
                        r_gyro = (N_mon/k0)**(1/Df) * r_mon_micron 

                        print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}      {Df:10.3f}              {r_gyro:10.5f}      {k0:10.3f}      {r_mon_micron:10.7f} ')
                        argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk','-mmf',str(r_mon_micron),str(Df),str(k0), '-q'] # MMF theory (aggregates): -c, link to refractive indices, -a, radius, -l, list of wavelengths (do whole file), -mmf, monomer radius r0, fractal dimension d_f, fractal prefactor k0
                else:
                        print(f'  {10000.0*radius[irad]:10.5f}         {rr_micron:10.5f}    {N_mon:10.0f}           JUST                   DOING            SINGLE       SPHERES ')
                        argument_list = ['-c','lnk_data/'+gas+'_VIRGA.lnk','-a', str(rr_micron),'-l','lnk_data/'+gas+'_VIRGA.lnk','-mie', '-q'] # Mie theory (single spherical particle)

                # run optool using the string created above - this finds kext[0], ksca[0], kabs[0], gsca[0], one particle radius at a time, for all wavelengths in the list, and stores them in a table within file "dustkappa.dat" (written in the optool directory)
                subprocess.run([f"{optool_dir}/optool"]+argument_list, cwd=optool_dir) # code executable + arguments, complete in the working directory of the compiled OPTOOL code 

                # read the wavelengths, k_abs, k_sca, and g_asymmetry values in the 'dustkappa.dat' file that was just created by optool
                arr=pd.read_csv(optool_dir+"/dustkappa.dat", names=['lambda','k_abs','k_sca','g_asymmetry'], delimiter=r"\s+", comment='#') # skip the comments in the header of the refractive index file, and read in the data
                arr= arr.drop([0,1]).reset_index(drop=True) # drop first two rows (data not needed) and re-index

                #print(arr)

                # read the density from the header of the material file
                density=pd.read_csv(optool_dir+'/lnk_data/'+gas+'_VIRGA.lnk',nrows=1, delimiter=r"\s+", comment='#', header=None) # read the line after the comments and before the data in the refractive index file (contains "num wavelengths, density" in that order)
                rho_p = density[1][0] # store the density of the condensate in g/cm3

                #convert k_abs, k_sca and g from optool output into Q values. Radius needs to be in centimeters here for this conversion since rho_p has g/cm3 units
                qe = (4 * (arr["k_abs"]+arr["k_sca"]) * rho_p * rr) / 3
                qs = (4 * arr["k_sca"] * rho_p * rr) / 3
                qa = (4 * arr["k_abs"] * rho_p * rr) / 3
                cos_qs = arr["g_asymmetry"] * qs

                #increment over the radius sub bins
                rr += dr5  
                rr_micron += dr5_micron
                
                #reshape the output arrays in the same format as output by PyMieScatt
                for iwave in range(nwave):
                    qext[iwave,irad] += qe[iwave]
                    qscat[iwave,irad] += qs[iwave]
                    cos_qscat[iwave,irad] += cos_qs[iwave]

        
        #smooth over all the radius bins 
        qext = qext / sub_radii  #matrix as a function of [iwave,irad] divide by 6 (the number of subradii bins)
        qscat = qscat / sub_radii  #matrix as a function of [iwave,irad]   
        cos_qscat = cos_qscat / sub_radii  #matrix as a function of [iwave,irad]   
                
    return qext, qscat, cos_qscat
