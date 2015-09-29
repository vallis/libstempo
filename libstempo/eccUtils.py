
"""
Created by stevertaylor and jellis18
Copyright (c) 2015 Stephen R. Taylor and Justin A. Ellis

Code developed by Stephen R. Taylor and incorporated into libsbempo by 
Justin A. Ellis. 

Relevant References are:

    Taylor et al. (2015) [http://adsabs.harvard.edu/abs/2015arXiv150506208T]
    Barack and Cutler (2004) [http://adsabs.harvard.edu/abs/2004PhRvD..69h2005B]

"""

from __future__ import division
import numpy as np
import scipy.special as ss
import scipy.constants as sc
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from pkg_resources import resource_filename, Requirement

SOLAR2S = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3
MPC2S = sc.parsec / sc.c * 1e6

def make_ecc_interpolant():
    """
    Make interpolation function from eccentricity file to
    determine number of harmonics to use for a given
    eccentricity.

    :returns: interpolant
    """
    pth = resource_filename(Requirement.parse('libstempo'),
                            'libstempo/ecc_vs_nharm.txt')

    fil = np.loadtxt(pth)

    return interp1d(fil[:,0], fil[:,1])



def get_edot(F, mc, e):
    """
    Compute eccentricity derivative from Taylor et al. (2015)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: de/dt

    """

    # chirp mass
    mc *= SOLAR2S

    dedt = -304/(15*mc) * (2*np.pi*mc*F)**(8/3) * e * \
        (1 + 121/304*e**2) / ((1-e**2)**(5/2))

    return dedt

def get_Fdot(F, mc, e):
    """
    Compute frequency derivative from Taylor et al. (2015)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: dF/dt

    """

    # chirp mass
    mc *= SOLAR2S

    dFdt = 48 / (5*np.pi*mc**2) * (2*np.pi*mc*F)**(11/3) * \
        (1 + 73/24*e**2 + 37/96*e**4) / ((1-e**2)**(7/2))

    return dFdt

def get_gammadot(F, mc, q, e):
    """
    Compute gamma dot from Barack and Cutler (2004)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param e: Eccentricity of binary

    :returns: dgamma/dt

    """

    # chirp mass
    mc *= SOLAR2S

    #total mass
    m = (((1+q)**2)/q)**(3/5) * mc

    dgdt = 6*np.pi*F * (2*np.pi*F*m)**(2/3) / (1-e**2) * \
        (1 + 0.25*(2*np.pi*F*m)**(2/3)/(1-e**2)*(26-15*e**2))

    return dgdt

def get_coupled_ecc_eqns(y, t, mc, q):
    """
    Computes the coupled system of differential
    equations from Peters (1964) and Barack &
    Cutler (2004). This is a system of three variables:
    
    F: Orbital frequency [Hz]
    e: Orbital eccentricity
    gamma: Angle of precession of periastron [rad]
    phase0: Orbital phase [rad]
    
    :param y: Vector of input parameters [F, e, gamma]
    :param t: Time [s]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    
    :returns: array of derivatives [dF/dt, de/dt, dgamma/dt, dphase/dt]
    """
    
    F = y[0]
    e = y[1]
    gamma = y[2]
    phase = y[3]
    
    #total mass
    m = (((1+q)**2)/q)**(3/5) * mc    
    
    dFdt = get_Fdot(F, mc, e)
    dedt = get_edot(F, mc, e)
    dgdt = get_gammadot(F, mc, q, e)
    dphasedt = 2*np.pi*F
     
    return np.array([dFdt, dedt, dgdt, dphasedt])

def solve_coupled_ecc_solution(F0, e0, gamma0, phase0, mc, q, t):
    """
    Compute the solution to the coupled system of equations
    from from Peters (1964) and Barack & Cutler (2004) at 
    a given time.
    
    :param F0: Initial orbital frequency [Hz]
    :param e0: Initial orbital eccentricity
    :param gamma0: Initial angle of precession of periastron [rad]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param t: Time at which to evaluate solution [s]
    
    :returns: (F(t), e(t), gamma(t), phase(t))
    
    """
    
    y0 = np.array([F0, e0, gamma0, phase0])

    y, infodict = odeint(get_coupled_ecc_eqns, y0, t, args=(mc,q), full_output=True)
    
    if infodict['message'] == 'Integration successful.':
        ret = y
    else:
        ret = 0
    
    return ret

def get_an(n, mc, dl, F, e):
    """
    Compute a_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: a_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S
    
    omega = 2 * np.pi * F
    
    amp = n * mc**(5/3) * omega**(2/3) / dl
    
    ret = -amp * (ss.jn(n-2,n*e) - 2*e*ss.jn(n-1,n*e) +
                  (2/n)*ss.jn(n,n*e) + 2*e*ss.jn(n+1,n*e) -
                  ss.jn(n+2,n*e))

    return ret

def get_bn(n, mc, dl, F, e):
    """
    Compute b_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: b_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S 
    
    omega = 2 * np.pi * F
    
    amp = n * mc**(5/3) * omega**(2/3) / dl
        
    ret = -amp * np.sqrt(1-e**2) *(ss.jn(n-2,n*e) - 2*ss.jn(n,n*e) +
                  ss.jn(n+2,n*e)) 

    return ret

def get_cn(n, mc, dl, F, e):
    """
    Compute c_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: c_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S
    
    omega = 2 * np.pi * F
    
    amp = 2 * mc**(5/3) * omega**(2/3) / dl
     
    ret = amp * ss.jn(n,n*e) / (n * omega)

    return ret

def calculate_splus_scross(nmax, mc, dl, F, e, t, l0, gamma, gammadot, inc):
    """
    Calculate splus and scross summed over all harmonics. 
    This waveform differs slightly from that in Taylor et al (2015) 
    in that it includes the time dependence of the advance of periastron.
    
    :param nmax: Total number of harmonics to use
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    :param t: TOAs [s]
    :param l0: Initial eccentric anomoly [rad]
    :param gamma: Angle of periastron advance [rad]
    :param gammadot: Time derivative of angle of periastron advance [rad/s]
    :param inc: Inclination angle [rad]

    """ 
    
    n = np.arange(1, nmax)

    # time dependent amplitudes
    an = get_an(n, mc, dl, F, e)
    bn = get_bn(n, mc, dl, F, e)
    cn = get_cn(n, mc, dl, F, e)

    # time dependent terms
    omega = 2*np.pi*F
    gt = gamma + gammadot * t
    lt = l0 + omega * t

    # tiled phase
    phase1 = n * np.tile(lt, (nmax-1,1)).T
    phase2 = np.tile(gt, (nmax-1,1)).T
    phasep = phase1 + 2*phase2
    phasem = phase1 - 2*phase2

    # intermediate terms
    sp = np.sin(phasem)/(n*omega-2*gammadot) + \
            np.sin(phasep)/(n*omega+2*gammadot)
    sm = np.sin(phasem)/(n*omega-2*gammadot) - \
            np.sin(phasep)/(n*omega+2*gammadot)
    cp = np.cos(phasem)/(n*omega-2*gammadot) + \
            np.cos(phasep)/(n*omega+2*gammadot)
    cm = np.cos(phasem)/(n*omega-2*gammadot) - \
            np.cos(phasep)/(n*omega+2*gammadot)
    

    splus_n = -0.5 * (1+np.cos(inc)**2) * (an*sp - bn*sm) + \
            (1-np.cos(inc)**2)*cn * np.sin(phase1)
    scross_n = np.cos(inc) * (an*cm - bn*cp)
        

    return np.sum(splus_n, axis=1), np.sum(scross_n, axis=1)
