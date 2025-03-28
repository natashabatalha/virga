import numpy as np
pi=np.pi
import miepython as mie
import os 

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


def calc_new_mieff(wave_in, nn,kk, radius, rup, fort_calc_mie = False):
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
            if irad== 0 :
                dr5= (( rup[0] - radius[0] ) / 5.)
                rr= radius[0]
            else:
                dr5 = ( rup[irad] - rup[irad-1] ) / 5.
                rr  = rup[irad-1]
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
    idummy, wave_in, nn, kk = np.loadtxt(open(filename,'rt').readlines(), unpack=True, usecols=[0,1,2,3])#[:-1]
    return wave_in,nn,kk


def get_r_grid(r_min=1e-5, n_radii=40):
    """
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
