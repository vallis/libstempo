
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
from scipy.integrate import odeint


def get_edot(F, mc, e):
    """
    Compute eccentricity derivative from Taylor et al. (2015)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: de/dt

    """

    # chirp mass
    mc *= 4.9e-6

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
    mc *= 4.9e-6

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
    mc *= 4.9e-6

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
    
    # chirp mass
    mc *= 4.9e-6
    
    #total mass
    m = (((1+q)**2)/q)**(3/5) * mc    
    
    dFdt = get_Fdot(F, mc, e)
    dedt = get_edot(F, mc, e)
    dgdt = get_gammadot(F, mc, q, e)
    dphasedt = F
     
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

def get_an(n, mc, dl, F, e, t, l0):
    """
    Compute a_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    :param t: Time [s]
    :param l0: Initial mean anomoly [rad]
    
    :returns: a_n
    
    """
    
    # convert to seconds
    mc *= 4.9e-6
    dl *= 1.0267e14
    
    omega = 2 * np.pi * F
    
    amp = mc**(5/3) / ( dl * omega**(1/3) )
    
    ret = -amp * ((ss.jn(n-2,n*e) - 2*e*ss.jn(n-1,n*e) +
                  (2/n)*ss.jn(n,n*e) + 2*e*ss.jn(n+1,n*e) -
                  ss.jn(n+2,n*e)) * np.sin(n*omega*t + n*l0))

    return ret

def get_bn(n, mc, dl, F, e, t, l0):
    """
    Compute b_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    :param t: Time [s]
    :param l0: Initial mean anomoly [rad]
    
    :returns: b_n
    
    """
    
    # convert to seconds
    mc *= 4.9e-6
    dl *= 1.0267e14
    
    omega = 2 * np.pi * F
    
    amp = mc**(5/3) / ( dl * omega**(1/3) )
    
    ret = amp * np.sqrt(1-e**2) *((ss.jn(n-2,n*e) - 2*ss.jn(n,n*e) +
                  ss.jn(n+2,n*e)) * np.cos(n*omega*t + n*l0))

    return ret

def get_cn(n, mc, dl, F, e, t, l0):
    """
    Compute c_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    :param t: Time [s]
    :param l0: Initial mean anomoly [rad]
    
    :returns: c_n
    
    """
    
    # convert to seconds
    mc *= 4.9e-6
    dl *= 1.0267e14
    
    omega = 2 * np.pi * F
    
    amp = mc**(5/3) / ( dl * omega**(1/3) )
    
    ret = amp * (2/n) * (ss.jn(n,n*e) * np.sin(n*omega*t + n*l0))

    return ret

def calculate_splus_scross(nmax, mc, dl, F, e, t, l0, gamma, gammadot, inc):
    
    for n in range(1, nmax):
        
        # time dependent amplitudes
        an = get_an(n, mc, dl, F, e, t, l0)
        bn = get_bn(n, mc, dl, F, e, t, l0)
        cn = get_cn(n, mc, dl, F, e, t, l0)

        # time dependent gamma
        gt = gamma + gammadot * t

        splus_n = -(1+np.cos(inc)**2) * (an * np.cos(2*gt) - bn * \
                                         np.sin(2*gt)) + (1-np.cos(inc)**2) * cn
        scross_n = 2 * np.cos(inc) * (bn * np.cos(2*gt) + an * np.sin(2*gt))

        yield splus_n, scross_n
