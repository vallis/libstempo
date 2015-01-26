
"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard), Justin Ellis (PAL/PAL2), and Chiara Mingarelli.

"""

import math
import matplotlib.pyplot as plt
import numpy as np
from math import factorial, sqrt, sin, cos, tan, acos, atan, pi, log
from cmath import exp
import scipy
from scipy.integrate import quad, dblquad
from scipy import special as sp
import random

norm = 3./(8*pi)
c00=sqrt(4*pi)

def calczeta(phi1, phi2, theta1, theta2):
    """
    Calculate the angular separation between position (phi1, theta1) and
    (phi2, theta2)
    """
    zeta = 0.0
    if phi1 == phi2 and theta1 == theta2:
        zeta = 0.0
    else:
        argument =sin(theta1)*sin(theta2)*cos(phi1-phi2) + cos(theta1)*cos(theta2)
        if argument < -1:
            zeta = np.pi
        elif argument > 1:
            zeta = 0.0
        else:
            zeta=acos(argument)
    ###
    return zeta


#################################
#################################

"""
Following functions taken from Gair et. al (2014),
involving solutions of integrals to define the ORF for an
arbitrarily anisotropic GW background.
"""


def Fminus00(qq, mm,ll,zeta):
    ##########
    integrand = 0.
   
    for ii in range(0,qq+1):
        for jj in range(mm,ll+1):
            integrand += ( (2.0**(ii-jj))*((-1.)**(qq-ii+jj+mm)) ) * (factorial(qq)*factorial(ll+jj) * ((2.0**(qq-ii+jj-mm+1)) - ((1.0+cos(zeta))**(qq-ii+jj-mm+1)))) / (factorial(ii)*factorial(qq-ii)*factorial(jj)*factorial(ll-jj)*factorial(jj-mm)*(qq-ii+jj-mm+1))

    return integrand


def Fminus01(qq, mm,ll,zeta):
    ##########
    integrand = 0.
    
    for ii in range(0,qq+1):
        for jj in range(mm,ll+1):
            integrand += ( (2.0**(ii-jj))*((-1.)**(qq-ii+jj+mm)) ) * (factorial(qq)*factorial(ll+jj) * ((2.0**(qq-ii+jj-mm+2)) - ((1.0+cos(zeta))**(qq-ii+jj-mm+2)))) / (factorial(ii)*factorial(qq-ii)*factorial(jj)*factorial(ll-jj)*factorial(jj-mm)*(qq-ii+jj-mm+2))

    return integrand

def Fplus01(qq, mm,ll,zeta):
    ##########
    integrand = 0.
    
    for ii in range(0,qq):
        for jj in range(mm,ll+1):
            integrand += ( (2.0**(ii-jj))*((-1.)**(ll+qq-ii+jj)) ) * (factorial(qq)*factorial(ll+jj) * ((2.0**(qq-ii+jj-mm)) - ((1.0-cos(zeta))**(qq-ii+jj-mm)))) / (factorial(ii)*factorial(qq-ii)*factorial(jj)*factorial(ll-jj)*factorial(jj-mm)*(qq-ii+jj-mm))

    if mm==ll:
        integrand += 0. 
    else:
        for jj in range(mm+1,ll+1):
            integrand += ( (2.0**(qq-jj))*((-1.)**(ll+jj)) ) * (factorial(ll+jj)*((2.0**(jj-mm)) - ((1.0-cos(zeta))**(jj-mm)))) / (factorial(jj)*factorial(ll-jj)*factorial(jj-mm)*(jj-mm))

    
    return integrand + ( ((-1.)**(ll+mm))*(2.0**(qq-mm))*factorial(ll+mm) * log(2./(1.0-cos(zeta))) ) / (1.0*factorial(mm)*factorial(ll-mm))

def Fplus00(qq, mm,ll,zeta):
    ##########
    integrand = 0.
    
    for ii in range(0,qq+1):
        for jj in range(mm,ll+1):
            integrand += ( (2.0**(ii-jj))*((-1.)**(ll+qq-ii+jj)) ) * (factorial(qq)*factorial(ll+jj) * ((2.0**(qq-ii+jj-mm+1)) - ((1.0-cos(zeta))**(qq-ii+jj-mm+1)))) / (factorial(ii)*factorial(qq-ii)*factorial(jj)*factorial(ll-jj)*factorial(jj-mm)*(qq-ii+jj-mm+1))

    return integrand



def ArbORF_calc(mm,ll,zeta):
    ##########
    if mm == 0:

        if ll>=0 and ll<=2:

            delta = [1.0+cos(zeta)/3., -(1.+cos(zeta))/3., 2.0*cos(zeta)/15.]
            if zeta==0.:
                return norm*0.5*sqrt( (2.0*ll+1.0)*pi ) * (delta[ll] - (1.0+cos(zeta))*Fminus00(0, 0,ll,zeta)) 
            else:
                return norm*0.5*sqrt( (2.0*ll+1.0)*pi ) * (delta[ll] - (1.0+cos(zeta))*Fminus00(0, 0,ll,zeta) - (1.0-cos(zeta))*Fplus01(1, 0,ll,zeta))

        else:
            if zeta==0.:
                return norm*0.5*sqrt( (2.0*ll+1.0)*pi ) * ( - (1.0+cos(zeta))*Fminus00(0, 0,ll,zeta)) 
            else:
                return norm*0.5*sqrt( (2.0*ll+1.0)*pi ) * ( - (1.0+cos(zeta))*Fminus00(0, 0,ll,zeta) - (1.0-cos(zeta))*Fplus01(1, 0,ll,zeta))

    elif mm == 1:

        if ll==1 or ll==2:

            delta = [2.0*sin(zeta)/3., -2.0*sin(zeta)/5.]
            return norm*0.25*sqrt( (2.0*ll+1.0)*pi )*sqrt( (1.0*factorial(ll-1))/(1.0*factorial(ll+1)) ) * (delta[ll-1] - (((1.0+cos(zeta))**(3./2.))/((1.0-cos(zeta))**(1./2.)))*Fminus00(1, 1,ll,zeta) - (((1.0-cos(zeta))**(3./2.))/((1.0+cos(zeta))**(1./2.)))*Fplus01(2, 1,ll,zeta))

        else:

            return norm*0.25*sqrt( (2.0*ll+1.0)*pi )*sqrt( (1.0*factorial(ll-1))/(1.0*factorial(ll+1)) ) * ( - (((1.0+cos(zeta))**(3./2.))/((1.0-cos(zeta))**(1./2.)))*Fminus00(1, 1,ll,zeta) - (((1.0-cos(zeta))**(3./2.))/((1.0+cos(zeta))**(1./2.)))*Fplus01(2, 1,ll,zeta))

    else:

        return -norm*0.25*sqrt( (2.0*ll+1.0)*pi )*sqrt( (1.0*factorial(ll-mm))/(1.0*factorial(ll+mm)) ) * ( (((1.0+cos(zeta))**(mm/2. + 1))/((1.0-cos(zeta))**(mm/2.)))*Fminus00(mm, mm,ll,zeta) - (((1.0+cos(zeta))**(mm/2.))/((1.0-cos(zeta))**(mm/2.-1.)))*Fminus01(mm-1, mm,ll,zeta) + (((1.0-cos(zeta))**(mm/2. + 1))/((1.0+cos(zeta))**(mm/2.)))*Fplus01(mm+1, mm,ll,zeta) - (((1.0-cos(zeta))**(mm/2.))/((1.0+cos(zeta))**(mm/2.-1.)))*Fplus00(mm, mm,ll,zeta) )



def dlmk(l,m,k,theta1):
    """
    returns value of d^l_mk as defined in allen, ottewill 97.
    Called by Dlmk
    """
    if m>=k:
        factor = sqrt(factorial(l-k)*factorial(l+m)/factorial(l+k)/factorial(l-m))
        part2 = (cos(theta1/2))**(2*l+k-m)*(-sin(theta1/2))**(m-k)/factorial(m-k)
        part3 = sp.hyp2f1(m-l,-k-l,m-k+1,-(tan(theta1/2))**2)
        return factor*part2*part3
    else:
        return (-1)**(m-k)*dlmk(l,k,m,theta1)

def Dlmk(l,m,k,phi1,phi2,theta1,theta2):
    """
    returns value of D^l_mk as defined in allen, ottewill 97.
    """
    return exp(complex(0.,-m*phi1))*dlmk(l,m,k,theta1)*exp(complex(0.,-k*gamma(phi1,phi2,theta1,theta2)))

def gamma(phi1,phi2,theta1,theta2):
    """
    calculate third rotation angle
    inputs are angles from 2 pulsars
    returns the angle.
    """
    if phi1 == phi2 and theta1 == theta2:
        gamma = 0
    else:
        gamma = atan( sin(theta2)*sin(phi2-phi1)/(cos(theta1)*sin(theta2)*cos(phi1-phi2) - sin(theta1)*cos(theta2)))
    if (cos(gamma)*cos(theta1)*sin(theta2)*cos(phi1-phi2) + sin(gamma)*sin(theta2)*sin(phi2-phi1) - cos(gamma)*sin(theta1)*cos(theta2)) >= 0:
        return gamma
    else:
        return pi + gamma



def ArbCompFrame_ORF(mm,ll,zeta):
    
    if zeta==0.:
        if ll>2: 
            return 0.
        elif ll==2:
            if mm==0: 
                return 2*0.25*norm*(4./3)*(sqrt(pi/5))*cos(zeta) # pulsar-term doubling
            else:
                return 0.
        elif ll==1:
            if mm==0: 
                return -2*0.5*norm*(sqrt(pi/3.))*(1.0+cos(zeta)) # pulsar-term doubling
            else:
                return 0.
        elif ll==0:
            return 2.0*norm*0.25*sqrt(pi*4)*(1+(cos(zeta)/3.)) # pulsar-term doubling
    elif zeta==pi:
        if ll>2: 
            return 0.
        elif ll==2 and mm!=0: 
            return 0.
        elif ll==1 and mm!=0: 
            return 0.
        else: 
            return ArbORF_calc(mm,ll,zeta)
    else:
        return ArbORF_calc(mm,ll,zeta)



def rotated_Gamma_ml(m,l,phi1,phi2,theta1,theta2,gamma_ml):
    """
    This function takes any gamma in the computational frame and rotates it to the
    cosmic frame. Special cases exist for dipole and qudrupole, as these have
    been found analytically.
    """
    rotated_gamma = 0
    for i in range(2*l+1):
        rotated_gamma += Dlmk(l,m,i-l,phi1,phi2,theta1,theta2).conjugate()*gamma_ml[i]
    return rotated_gamma

def real_rotated_Gammas(m,l,phi1,phi2,theta1,theta2,gamma_ml):
    """
    This function returns the real-valued form of the Overlap Reduction Functions,
    see Eqs 47 in Mingarelli et al, 2013.
    """
    if m>0:
        ans=(1./sqrt(2))*(rotated_Gamma_ml(m,l,phi1,phi2,theta1,theta2,gamma_ml)+(-1)**m*rotated_Gamma_ml(-m,l,phi1,phi2,theta1,theta2,gamma_ml))
        return ans.real
    if m==0:
        return rotated_Gamma_ml(0,l,phi1,phi2,theta1,theta2,gamma_ml).real
    if m<0:
        ans=(1./sqrt(2)/complex(0.,1))*(rotated_Gamma_ml(-m,l,phi1,phi2,theta1,theta2,gamma_ml)-(-1)**m*rotated_Gamma_ml(m,l,phi1,phi2,theta1,theta2,gamma_ml))
        return ans.real



####################################
####################################


def AnisCoeff(psr_locs, lmax):
    corr=[]
    for ll in range(1,lmax+1):  # RANDOM BUG MEANS WE HAVE TO START WITH l=1 AND FILL THE HELLINGS-DOWNS CURVE SEPARATELY
        mmodes = 2*ll+1     # Number of modes for this ll
        for mm in range(mmodes):
            corr.append(np.zeros((len(psr_locs), len(psr_locs))))
            
        for aa in range(len(psr_locs)):
            for bb in range(aa, len(psr_locs)):
                rot_Gs=[]
                plus_gamma_ml = [] #this will hold the list of gammas evaluated at a specific value of phi1,2, and theta1,2.
                neg_gamma_ml = []
                gamma_ml = []
                
                #pre-calculate all the gammas so this gets done only once. Need all the values to execute rotation codes.
                for mm in range(ll+1):
                    zeta = calczeta(psr_locs[:,0][aa],psr_locs[:,0][bb],psr_locs[:,1][aa],psr_locs[:,1][bb])
                    
                    intg_gamma=ArbCompFrame_ORF(mm,ll,zeta)
                    neg_intg_gamma=(-1)**(mm)*intg_gamma  # just (-1)^m Gamma_ml since this is in the computational frame
                    plus_gamma_ml.append(intg_gamma)     #all of the gammas from Gamma^-m_l --> Gamma ^m_l
                    neg_gamma_ml.append(neg_intg_gamma)  #get the neg m values via complex conjugates 
                    
                neg_gamma_ml=neg_gamma_ml[1:]            #this makes sure we don't have 0 twice
                rev_neg_gamma_ml=neg_gamma_ml[::-1]      #reverse direction of list, now runs from -m .. 0
                gamma_ml=rev_neg_gamma_ml+plus_gamma_ml
                
                mindex = len(corr) - mmodes
                for mm in range(mmodes):
                    m = mm - ll
                    
                    corr[mindex+mm][aa, bb] = real_rotated_Gammas(m, ll, psr_locs[:,0][aa], psr_locs[:,0][bb], psr_locs[:,1][aa], psr_locs[:,1][bb], gamma_ml)
                    
                    if aa != bb:
                        corr[mindex+mm][bb, aa] = corr[mindex+mm][aa, bb]
                        
                        
    return corr



def CorrBasis(psr_locs,lmax):
    
    positions = np.array([np.array([np.sin(psr_locs[i][1])*np.cos(psr_locs[i][0]),np.sin(psr_locs[i][1])*np.sin(psr_locs[i][0]),np.cos(psr_locs[i][1])]) for i in range(len(psr_locs))])

    AngSepGrid = np.zeros((len(positions),len(positions)))
    for i in range(len(positions)):
        for j in range(len(positions)):
            AngSepGrid[i][j] = np.dot(positions[i],positions[j])

    xsep = 0.5*(1.-AngSepGrid)
    np.fill_diagonal(xsep,np.ones(len(positions))) # stops errors popping up
    HD = 1.5*xsep*np.log(xsep) - 0.25*xsep + 0.5 + 0.5*np.eye(len(positions))

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i==j:
                HD[i][j] = 1./np.sqrt(4.0*np.pi)
            else:
                HD[i][j] = HD[i][j]/np.sqrt(4.0*np.pi)

    basis = [HD]

    if lmax > 0:
        basis.extend(AnisCoeff(psr_locs, lmax))

    return basis
