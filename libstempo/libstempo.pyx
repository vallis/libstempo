#cython: language_level=3

import os, sys, math, re, time
from distutils.version import StrictVersion

import collections

try:
    from collections import OrderedDict
except ImportError:
    # this is for Python 2.6 compatibility, we may be able to drop it
    from ordereddict import OrderedDict

# Python 2/3 compatibility

if sys.version_info[0] < 3:
    from itertools import izip as zip

    string = lambda s: s
    string_dtype = 'S'
else:
    # what is the default encoding here?
    string = lambda s: s.decode()
    string_dtype = 'U'

# get zip-as-iterator behavior in Python 2
try:
    import itertools.izip as zip
except ImportError:
    pass

from libc cimport stdlib, stdio
from libc.string cimport strncpy, memset

from cython cimport view

import numpy
cimport numpy

import scipy.linalg

try:
    import astropy.units as u
    from astropy.units import Quantity
    from astropy.time import Time
    import astropy.constants
    try:
        import astropy.erfa as erfa
    except ImportError:
        import astropy._erfa as erfa

    # missing several parameters; units are discussed in tempo2/initialize.C

    lts = astropy.units.def_unit(['lightsecond','ls','lts'],astropy.constants.c * u.s)

    map_units = {
                 'F0': u.Hz,'F1': u.Hz/u.s,'F2': u.Hz/u.s**2,
                 'RAJ': u.rad,'DECJ': u.rad,'ELONG': u.deg,'ELAT': u.deg,
                 'PMRA': u.mas / u.yr,'PMDEC': u.mas / u.yr,'PMELONG': u.mas / u.yr,'PMELAT': u.mas / u.yr,
                 'PX': u.mas,
                 'PB': u.d,'ECC': u.dimensionless_unscaled,'A1': lts,'OM': u.deg,
                 'EPS1': u.dimensionless_unscaled,'EPS2': u.dimensionless_unscaled,
                 # KOM, KIN?
                 'SHAPMAX': u.dimensionless_unscaled,'OMDOT': u.deg/u.yr,
                 # PBDOT?
                 'ECCDOT': 1/u.s,'A1DOT': lts/u.s,'GAMMA': u.s,
                 # XPBDOT?
                 # EPS1DOT, EPS2DOT?
                 'MTOT': u.Msun,'M2': u.Msun,
                 # DTHETA, XOMDOT
                 'SIN1': u.dimensionless_unscaled,
                 # DR, A0, B0, BP, BPP, AFAC
                 'DM': u.cm**-3 * u.pc,'DM1': u.cm**-3 * u.pc * u.yr**-1, # how many should we do?
                 'POSEPOCH': u.day,'T0': u.day,'TASC': u.day
                 }

    map_times = ['POSEPOCH','T0','TASC']
except:
    print("Warning: cannot find astropy, units support will not be available.")

from functools import wraps

from . import utils

# return numpy array as astropy table with unit
def dimensionfy(unit):
    def dimensionfy_decorator(func):
        def dimensionfy_wrapper(*args,**kwargs):
            array = func(*args,**kwargs)

            if args[0].units:
                return Quantity(array,unit=unit,copy=False)
            else:
                return array
        return dimensionfy_wrapper
    return dimensionfy_decorator

cdef extern from "GWsim-stub.h":
    cdef bint HAVE_GWSIM

    ctypedef struct gwSrc:
        long double theta_g
        long double phi_g
        long double omega_g
        long double phi_polar_g

    void GWbackground(gwSrc *gw,int numberGW,long *idum,long double flo,long double fhi,double gwAmp,double alpha,int loglin)
    void GWdipolebackground(gwSrc *gw,int numberGW,long *idum,long double flo,long double fhi, double gwAmp,double alpha,int loglin, double *dipoleamps)
    void setupGW(gwSrc *gw)
    void setupPulsar_GWsim(long double ra_p,long double dec_p,long double *kp)
    long double calculateResidualGW(long double *kp,gwSrc *gw,long double obstime,long double dist)

cdef extern from "tempo2.h":
    enum: MAX_PSR_VAL
    enum: MAX_FILELEN
    enum: MAX_OBSN_VAL
    enum: MAX_PARAMS
    enum: MAX_JUMPS
    enum: MAX_FLAGS
    enum: MAX_FLAG_LEN
    enum: MAX_FIT
    enum: T2C_TEMPO
    enum: T2C_IAU2000B
    enum: REFPHS_MEAN
    enum: REFPHS_TZR
    enum: param_pepoch
    enum: param_raj
    enum: param_decj
    enum: param_LAST
    enum: param_ZERO
    enum: param_JUMP
    enum: MAX_T2EFAC
    enum: MAX_T2EQUAD
    enum: MAX_TNEF
    enum: MAX_TNEQ
    enum: MAX_TNSQ
    enum: MAX_TNECORR

    cdef char *TEMPO2_VERSION "TEMPO2_h_VER"

    int MAX_PSR, MAX_OBSN

    ctypedef struct parameter:
        char **label
        char **shortlabel
        long double *val
        long double *err
        int  *fitFlag
        int  *paramSet
        long double *prefit
        long double *prefitErr
        int aSize

    ctypedef struct observation:
        long double sat        # site arrival time
        long double origsat	   # Backup of SAT
        long double sat_day	   # Just the Day part
        long double sat_sec	   # Just the Sec part
        long double bat        # barycentric arrival time
        long double bbat       # barycentric arrival time
        long double batCorr    #update from sat-> bat
        long double pet        # pulsar emission time
        int clockCorr          # = 1 for clock corrections to be applied, = 0 for BAT
        int delayCorr          # = 1 for time delay corrections to be applied, = 0 for BAT
        int deleted            # 1 if observation deleted, -1 if not in fit
        long double prefitResidual
        long double residual
        double toaErr          # error on TOA (in us)
        double origErr         # original error on TOA after reading tim file (in us)
        double toaDMErr        # error on TOA due to DM (in us)
        char **flagID          # ID of flags
        char **flagVal         # Value of flags
        int nFlags             # Number of flags set
        double freq            # frequency of observation (in MHz)
        double freqSSB         # frequency of observation in barycentric frame (in Hz)
        char fname[MAX_FILELEN] # name of datafile giving TOA
        char telID[100]        # telescope ID
        double sun_ssb[6]      # Sun wrt SSB
        double earth_ssb[6]    # Earth center wrt SSB
        double planet_ssb[9][6]    # Planet centers wrt SSB
        double observatory_earth[6]    # Obs wrt Earth center
        double psrPos[3]       # Unit vector to the pulsar position
        double zenith[3]       # Zenith vector, in BC frame. Length=geodetic height
        long double torb       # Combined binary delay
        long long pulseN       # Pulse number
        long double roemer     # Roemer delay
        double shapiroDelaySun     # Shapiro delay caused by the Sun
        double phaseOffset     # Phase offset
        long double phase      # the phase (cycles)
        double efac            # Error multiplication factor
        double equad           # Value to add in quadrature
        int jump[MAX_FLAGS]    # Jump region
        int obsNjump           # Number of jumps for this observation
        int fdjump[MAX_FLAGS]
        int obsNfdjump

    ctypedef int param_label

    ctypedef double (*paramDerivFunc)(pulsar*,int,double,int,param_label,int)
    ctypedef void (*paramUpdateFunc)(pulsar*,int,param_label,int,double,double)

    ctypedef struct FitInfo:
        unsigned int nParams
        unsigned int nConstraints
        int paramIndex[MAX_FIT]
        int paramCounters[MAX_FIT]
        paramDerivFunc paramDerivs[MAX_FIT]
        paramUpdateFunc updateFunctions[MAX_FIT]

    ctypedef struct observatory:
        double height_grs80     # GRS80 geodetic height

    ctypedef struct pulsar:
        parameter param[MAX_PARAMS]
        observation *obsn
        char *name
        int nobs
        int rescaleErrChisq
        int noWarnings
        double fitChisq
        int nJumps
        double jumpVal[MAX_JUMPS]
        int fitJump[MAX_JUMPS]
        double jumpValErr[MAX_JUMPS]
        char *binaryModel
        int t2cMethod               # How to transform from terrestrial to celestial coords. Set in parfile with T2CMETHOD
                                    # tempo2 supports T2C_IAU2000B (default) and T2C_TEMPO
        char *JPL_EPHEMERIS
        char *ephemeris
        int useCalceph
        int eclCoord            # = 1 for ecliptic coords otherwise celestial coords
        double posPulsar[3]     # 3-unitvector pointing at the pulsar
        # long double phaseJump[MAX_JUMPS] # Time of phase jump (Deprecated. WHY?)
        int phaseJumpID[MAX_JUMPS]         # ID of closest point to phase jump
        int phaseJumpDir[MAX_JUMPS]        # Size and direction of phase jump
        int nPhaseJump                     # Number of phase jumps
        double rmsPost
        char clock[16]
        FitInfo fitinfo

        # noise parameters follow

        # T2EFAC
        int nT2efac
        char T2efacFlagID[MAX_T2EFAC][MAX_FLAG_LEN]
        char T2efacFlagVal[MAX_T2EFAC][MAX_FLAG_LEN]
        double T2efacVal[MAX_T2EFAC]

        # GLOBAL_EFAC in timfile???
        double T2globalEfac

        # T2EQUAD
        int nT2equad
        char T2equadFlagID[MAX_T2EQUAD][MAX_FLAG_LEN]
        char T2equadFlagVal[MAX_T2EQUAD][MAX_FLAG_LEN]
        double T2equadVal[MAX_T2EQUAD]

        # TNEF
        int nTNEF
        char TNEFFlagID[MAX_TNEF][MAX_FLAG_LEN]
        char TNEFFlagVal[MAX_TNEF][MAX_FLAG_LEN]
        double TNEFVal[MAX_TNEF]

        # TNGlobalEF
        double TNGlobalEF

        # TNEQ
        int nTNEQ
        char TNEQFlagID[MAX_TNEQ][MAX_FLAG_LEN]
        char TNEQFlagVal[MAX_TNEQ][MAX_FLAG_LEN]
        double TNEQVal[MAX_TNEQ]

        # TNGlobalEQ
        double TNGlobalEQ

        # addTNGlobalEQ
        double addTNGlobalEQ

        # TNSQ ???
        int nTNSQ
        char TNSQFlagID[MAX_TNSQ][MAX_FLAG_LEN]
        char TNSQFlagVal[MAX_TNSQ][MAX_FLAG_LEN]
        double TNSQVal[MAX_TNSQ]

        # ECORR/TNECORR
        int nTNECORR
        char TNECORRFlagID[MAX_TNECORR][MAX_FLAG_LEN]
        char TNECORRFlagVal[MAX_TNECORR][MAX_FLAG_LEN]
        double TNECORRVal[MAX_TNECORR]

        # TNRedAmp and TNRedGam (RNAMP and RNIDX are converted in readParfile.C)
        double TNRedAmp
        double TNRedGam
        int TNRedC

        # set reference observation
        char refphs
        char tzrsite[100]

    void initialise(pulsar *psr, int noWarnings)
    void destroyOne(pulsar *psr)

    void readParfile(pulsar *psr,char parFile[][MAX_FILELEN],char timFile[][MAX_FILELEN],int npsr)
    void readTimfile(pulsar *psr,char timFile[][MAX_FILELEN],int npsr)

    void preProcess(pulsar *psr,int npsr,int argc,char *argv[])
    void formBatsAll(pulsar *psr,int npsr)
    void updateBatsAll(pulsar *psr,int npsr)                    # what's the difference?
    void formResiduals(pulsar *psr,int npsr,int removeMean)

    # void doFit(pulsar *psr,int npsr,int writeModel) --- obsoleted
    # void doFitAll(pulsar *psr,int npsr, const char *covarFuncFile) --- obsoleted
    # void FITfuncs(double x,double afunc[],int ma,pulsar *psr,int ipos) -- obsoleted
    # void FITfuncs(double x,double afunc[],int ma,pulsar *psr,int ipos,int ipsr) --- obsoleted
    # void updateParameters(pulsar *psr,int p,double val[],double error[]) -- obsoleted

    int turn_hms(double turn,char *hms)

    void textOutput(pulsar *psr,int npsr,double globalParameter,int nGlobal,int outRes,int newpar,char *fname)
    void writeTim(char *timname,pulsar *psr,char *fileFormat)

    observatory *getObservatory(char *code)

cdef extern from "t2fit-stub.h":
    double t2FitFunc_jump(pulsar *psr,int ipsr,double x,int ipos,param_label label,int k)
    void t2UpdateFunc_jump(pulsar *psr,int ipsr,param_label label,int k,double val,double err)

    double t2FitFunc_zero(pulsar *psr,int ipsr,double x,int ipos,param_label label,int k)
    void t2UpdateFunc_zero(pulsar *psr,int ipsr,param_label label,int k,double val,double err)

    void t2fit_fillOneParameterFitInfo(pulsar* psr,param_label fit_param,const int k,FitInfo& OUT)

cdef void set_longdouble_from_array(long double *p,numpy.ndarray[numpy.npy_longdouble,ndim=0] a):
    p[0] = (<long double*>(a.data))[0]

cdef void set_longdouble(long double *p,object o):
    if isinstance(o,numpy.longdouble):
        set_longdouble_from_array(p,o[...])
    elif isinstance(o,numpy.ndarray) and o.dtype == numpy.longdouble and o.ndim == 0:
        set_longdouble_from_array(p,o)
    else:
        p[0] = o

cdef object get_longdouble_as_scalar(long double v):
    cdef numpy.ndarray ret = numpy.array(0,dtype=numpy.longdouble)
    (<long double*>ret.data)[0] = v
    return ret.item()

cdef class tempopar:
    cdef public object name

    cdef public object unit
    cdef public object timescale

    cdef public int ct
    cdef public int subct

    cdef int _isjump
    cdef void *_val
    cdef void *_err
    cdef int *_fitFlag
    cdef int *_paramSet

    def __init__(self,*args,**kwargs):
        raise TypeError("This class cannot be instantiated from Python.")

    def _unitify(self,val):
        if self.unit:
            return Quantity(val,unit=self.unit)
        else:
            return val

    property val:
        def __get__(self):
            if not self._isjump:
                return self._unitify(get_longdouble_as_scalar((<long double*>self._val)[0]))
            else:
                return self._unitify(float((<double*>self._val)[0]))

        def __set__(self,value):
            if self.timescale and isinstance(value,Time):
                time = getattr(value,self.timescale)
                value = Quantity(numpy.longdouble(time.jd1 - erfa.DJM0) + numpy.longdouble(time.jd2),
                                 unit=u.day)
                value = value.to(self.unit).value
            elif self.unit and isinstance(value,Quantity):
                value = value.to(self.unit).value

            if not self._isjump:
                if not self._paramSet[0]:
                    self._paramSet[0] = 1

                set_longdouble(<long double*>self._val,value)
            else:
                (<double*>self._val)[0] = value

    property err:
        def __get__(self):
            if not self._isjump:
                return self._unitify(get_longdouble_as_scalar((<long double*>self._err)[0]))
            else:
                return self._unitify(float((<double*>self._err)[0]))

        def __set__(self,value):
            if self.unit and isinstance(value,Quantity):
                value = value.to(self.unit).value

            if not self._isjump:
                set_longdouble(<long double*>self._err,value)
            else:
                (<double*>self._err)[0] = value

    property fit:
        def __get__(self):
            return True if self._fitFlag[0] else False

        def __set__(self,value):
            if value:
                if not self._isjump and not self._paramSet[0]:
                    self._paramSet[0] = 1

                self._fitFlag[0] = 1
            else:
                self._fitFlag[0] = 0

    # note that paramSet is not always respected in tempo2
    property set:
        def __get__(self):
            if not self._isjump:
                return True if self._paramSet[0] else False
            else:
                return True

        def __set__(self,value):
            if not self._isjump:
                if value:
                    self._paramSet[0] = 1
                else:
                    self._paramSet[0] = 0
            elif not value:
                raise ValueError("JUMP parameters declared in the par file cannot be unset in tempo2.")

    property isjump:
        def __get__(self):
            return True if self._isjump else False

    def __str__(self):
        if self.set:
            return 'tempo2 parameter %s (%s): %s +/- %s' % (self.name,'fitted' if self.fit else 'not fitted',repr(self.val),repr(self.err))
        else:
            return 'tempo2 parameter %s (unset)'

# since the __init__ for extension classes must have a Python signature,
# we use a factory function to initialize its attributes to pure-C objects

map_coords = {'RAJ': 'ELONG', 'DECJ': 'ELAT', 'PMRA': 'PMELONG', 'PMDEC': 'PMELAT'}

cdef create_tempopar(parameter par,int ct,int subct,int eclCoord,object units):
    cdef tempopar newpar = tempopar.__new__(tempopar)

    try:
        newpar.name = string(par.shortlabel[subct])
    except UnicodeDecodeError:
        newpar.name = ''

    if newpar.name in ['RAJ','DECJ','PMRA','PMDEC'] and eclCoord == 1:
        newpar.name = map_coords[newpar.name]

    if units:
        newpar.unit = map_units.get(newpar.name,u.dimensionless_unscaled)
        # TO DO: need to find out what tempo2 is using
        newpar.timescale = 'tcb' if newpar.name in map_times else None
    else:
        newpar.unit = None
        newpar.timescale = None

    newpar._isjump = 0

    newpar._val = &par.val[subct]
    newpar._err = &par.err[subct]
    newpar._fitFlag = &par.fitFlag[subct]
    newpar._paramSet = &par.paramSet[subct]

    newpar.ct = ct
    newpar.subct = subct

    return newpar

# TODO: note that currently we cannot change the number of jumps programmatically
cdef create_tempojump(pulsar *psr,int ct,object units):
    cdef tempopar newpar = tempopar.__new__(tempopar)

    # TO DO: proper units
    if units:
        newpar.unit = u.dimensionless_unscaled
        newpar.timescale = None
    else:
        newpar.unit = None
        newpar.timescale = None

    newpar.name = 'JUMP{0}'.format(ct)

    newpar._isjump = 1

    newpar._val = &psr.jumpVal[ct]
    newpar._err = &psr.jumpValErr[ct]
    newpar._fitFlag = &psr.fitJump[ct]

    newpar.ct = param_JUMP
    newpar.subct = ct

    return newpar


# TODO: check if consistent with new API
cdef class GWB:
    cdef gwSrc *gw
    cdef int ngw

    def __cinit__(self,ngw=1000,seed=None,flow=1e-8,fhigh=1e-5,gwAmp=1e-20,alpha=-0.66,logspacing=True, \
                  dipoleamps=None,dipoledir=None,dipolemag=None):
        self.gw = <gwSrc *>stdlib.malloc(sizeof(gwSrc)*ngw)
        self.ngw = ngw

        is_dipole = False
        is_anis = False

        if seed is None:
            seed = -int(time.time())

        gwAmp = gwAmp * (86400.0*365.25)**alpha

        cdef long idum = seed
        cdef numpy.ndarray[double,ndim=1] dipamps = numpy.zeros(3,numpy.double)

        if dipoleamps is not None:
            dipoleamps = numpy.array(dipoleamps/(4.0*numpy.pi))
            if numpy.sum(dipoleamps**2) > 1.0:
                raise ValueError("Full dipole amplitude > 1. Change the amplitudes")

            dipamps[:] = dipoleamps[:]
            is_dipole = True

        if dipoledir is not None and dipolemag is not None:
            dipolemag/=4.0*numpy.pi
            dipamps[0]=numpy.cos(dipoledir[1])*dipolemag
            dipamps[1]=numpy.sin(dipoledir[1])*numpy.cos(dipoledir[0])*dipolemag
            dipamps[2]=numpy.sin(dipoledir[1])*numpy.sin(dipoledir[0])*dipolemag

            is_dipole = True

        if is_dipole:
            dipamps = numpy.ascontiguousarray(dipamps, dtype=numpy.double)
            if not HAVE_GWSIM:
                raise NotImplementedError("libstempo was compiled against an older tempo2 that does not implement GWdipolebackground.")
            GWdipolebackground(self.gw,ngw,&idum,flow,fhigh,gwAmp,alpha,1 if logspacing else 0, &dipamps[0])
        else:
            GWbackground(self.gw,ngw,&idum,flow,fhigh,gwAmp,alpha,1 if logspacing else 0)

        for i in range(ngw):
          setupGW(&self.gw[i])

    def __dealloc__(self):
        stdlib.free(self.gw)

    def gwb_sig(self,tempopulsar pulsar, distance=1):
        cdef long double dist = distance * 3.086e19

        cdef long double ra_p  = pulsar.psr[0].param[param_raj].val[0]
        cdef long double dec_p = pulsar.psr[0].param[param_decj].val[0]

        cdef long double epoch = pulsar.psr[0].param[param_pepoch].val[0]

        cdef long double kp[3]

        setupPulsar_GWsim(ra_p,dec_p,&kp[0])

        cdef numpy.ndarray[long double,ndim=1] res = numpy.zeros(pulsar.nobs,numpy.longdouble)
        cdef long double obstime

        for i in range(pulsar.nobs):
            obstime = (pulsar.psr[0].obsn[i].sat - epoch)*86400.0
            res[i] = 0.0

            for k in range(self.ngw):
                res[i] = res[i] + calculateResidualGW(kp,&self.gw[k],obstime,dist)

        res[:] = res[:] - numpy.mean(res)

        return res

    def add_gwb(self,tempopulsar pulsar,distance=1):
        pulsar.stoas[:] += self.gwb_sig(pulsar, distance) / 86400.0

    def gw_dist(self):
        theta = numpy.zeros(self.ngw)
        phi = numpy.zeros(self.ngw)
        omega = numpy.zeros(self.ngw)
        polarization = numpy.zeros(self.ngw)

        for i in range(self.ngw):
            theta[i] = self.gw[i].theta_g
            phi[i] = self.gw[i].phi_g
            omega[i] = self.gw[i].omega_g
            polarization[i] = self.gw[i].phi_polar_g

        return theta, phi, omega, polarization

noisepar = collections.namedtuple('noisepar',['val','flag','flagval'])

def parse_tempo2version(s):
    """Tempo2 now temporarily has a different version system, so convert"""
    prog = re.compile("[0-9]+\.[0-9]+\.[0-9]+")
    if prog.match(string(s)):
        return string(s)
    else:
        return string(s).split()[1]

def tempo2version():
    return StrictVersion(parse_tempo2version(TEMPO2_VERSION))

# this is a Cython extension class; the benefit is that it can hold C attributes,
# but all attributes must be defined in the code

cdef class tempopulsar:
    """tempopulsar(parfile, timfile=None, warnings=False, fixprefiterrors=True,
                   dofit=False, maxobs=None, units=False, ephem=None, t2cmethod=None,
                   toas=None, toaerrs=None, observatory=None, obsfreq=1400)"""

    cpdef public object parfile
    cpdef public object timfile

    cpdef public object units

    cdef int npsr           # number of pulsars
    cdef pulsar *psr        # array of pulsar structures

    cpdef object pardict    # dictionary of parameter proxies

    cpdef int nobs_         # number of observations

    cpdef object flagnames_ # a list of all flags that have values
    cpdef object flags_     # a dictionary of numpy arrays with flag values

    cpdef public object noisemodel

    cpdef object __input_toas  # input TOAs
    cpdef object __input_toaerrs  # input TOA errors (us)
    cpdef object __input_observatory  # input observatories
    cpdef object __input_obsfreq   # input observation frequencies

    # TO DO: is cpdef required here?
    cpdef jumpval, jumperr

    def __cinit__(self, parfile, timfile=None, warnings=False,
                  fixprefiterrors=True, dofit=False, maxobs=None,
                  units=False, ephem=None, clk=None, t2cmethod=None,
                  toas=None, toaerrs=None, observatory=None,
                  obsfreq=1440):

        # initialize

        global MAX_PSR, MAX_OBSN

        self.npsr = 1

        self.__input_toas = toas
        self.__input_toaerrs = toaerrs
        self.__input_observatory = observatory
        self.__input_obsfreq = obsfreq

        # to save memory, only allocate space for this many pulsars and observations
        MAX_PSR = 1
        try:
            MAX_OBSN = len(toas) + 1  # add spare in case needed for reference obs
        except TypeError:
            MAX_OBSN = MAX_OBSN_VAL if maxobs is None else maxobs 

        self.psr = <pulsar *>stdlib.malloc(sizeof(pulsar)*MAX_PSR)
        initialise(self.psr,1)          # 1 for no warnings

        # read par and tim file

        # tim rewriting is not needed with tempo2/readTimfile.C >= 1.22 (date: 2014/06/12 02:25:54),
        # which follows relative paths; closest tempo2.h version is 1.90 (date: 2014/06/24 20:03:34)
        if tempo2version() >= StrictVersion("1.90"):
            self._readfiles(parfile,timfile)
        else:
            timfile = rewritetim(timfile)
            self._readfiles(parfile,timfile)
            os.unlink(timfile)

        # set tempo2 flags

        self.psr.rescaleErrChisq = 0        # do not rescale fit errors by sqrt(red. chisq)

        if not warnings:
            self.psr.noWarnings = 2         # do not show some warnings

        # set clock if given
        if clk is not None:
            self.clock = clk

        # set conversion from terrestrial to celestial
        if t2cmethod is not None:
            if t2cmethod == 'TEMPO':
                self.psr[0].t2cMethod = T2C_TEMPO
            elif t2cmethod == 'IAU2000B':
                self.psr[0].t2cMethod = T2C_IAU2000B
            else:
                raise ValueError("Unknown T2CMETHOD '{}'.".format(t2cmethod))

        # set ephemeris if given
        if ephem is not None:
            self.ephemeris = ephem

        # preprocess the data

        preProcess(self.psr,self.npsr,0,NULL)
        formBatsAll(self.psr,self.npsr)

        # create parameter proxies

        self.units = units

        self.nobs_ = self.psr[0].nobs
        self._readpars(fixprefiterrors=fixprefiterrors)
        self._readflags()

        # do a fit if requested
        if dofit:
            self.fit()

    def __dealloc__(self):
        for i in range(self.npsr):
            destroyOne(&(self.psr[i]))
            stdlib.free(&(self.psr[i]))

    def _readfiles(self,parfile,timfile=None):
        cdef char parFile[MAX_PSR_VAL][MAX_FILELEN]
        cdef char timFile[MAX_PSR_VAL][MAX_FILELEN]

        if timfile is None:
            timfile = re.sub('\.par$','.tim',parfile)

        self.parfile = parfile
        self.timfile = timfile

        if not os.path.isfile(parfile):
            raise IOError("Cannot find parfile {0}.".format(parfile))

        if not os.path.isfile(timfile) and self.__input_toas is None:
            # hail Mary pass
            maybe = '../tim/{0}'.format(timfile)
            if os.path.isfile(maybe):
                timfile = maybe
            else:
                raise IOError("Cannot find timfile {0}.".format(timfile))

        parfile_bytes, timfile_bytes = parfile.encode(), timfile.encode()

        for checkfile in [parfile_bytes,timfile_bytes]:
            if len(checkfile) > MAX_FILELEN - 1:
                raise IOError("Filename {0} is too long for tempo2.".format(checkfile))

        stdio.sprintf(parFile[0],"%s",<char *>parfile_bytes)
        stdio.sprintf(timFile[0],"%s",<char *>timfile_bytes)

        readParfile(self.psr,parFile,timFile,self.npsr)   # load the parameters    (all pulsars)

        # save noise parameters and reset them in the pulsar structure
        # so the data is not touched

        self.noisemodel = OrderedDict()

        for i in range(self.psr[0].nT2efac):
            name = 'efac_' + string(self.psr[0].T2efacFlagVal[i])

            self.noisemodel[name] = noisepar(val=self.psr[0].T2efacVal[i],
                                             flag=string(&self.psr[0].T2efacFlagID[i][1]),
                                             flagval=string(self.psr[0].T2efacFlagVal[i]))

        self.psr[0].nT2efac = 0

        for i in range(self.psr[0].nT2equad):
            efacname = 'efac_' + string(self.psr[0].T2equadFlagVal[i])
            name = 'equad_' + string(self.psr[0].T2equadFlagVal[i])

            value = self.psr[0].T2equadVal[i] * (self.noisemodel[efacname].val if efacname in self.noisemodel else 1.0)

            self.noisemodel[name] = noisepar(val=value,
                                             flag=string(&self.psr[0].T2equadFlagID[i][1]),
                                             flagval=string(self.psr[0].T2equadFlagVal[i]))

        self.psr[0].nT2equad = 0

        # should check for conflicts between T2 and TN values?

        for i in range(self.psr[0].nTNEF):
            name = 'efac_' + string(self.psr[0].TNEFFlagVal[i])

            self.noisemodel[name] = noisepar(val=self.psr[0].TNEFVal[i],
                                             flag=string(&self.psr[0].TNEFFlagID[i][1]),
                                             flagval=string(self.psr[0].TNEFFlagVal[i]))

        for i in range(self.psr[0].nTNEQ):
            name = 'equad_' + string(self.psr[0].TNEQFlagVal[i])

            value = 1e6 * 10**self.psr[0].TNEQVal[i]

            self.noisemodel[name] = noisepar(val=value,
                                             flag=string(&self.psr[0].TNEQFlagID[i][1]),
                                             flagval=string(self.psr[0].TNEQFlagVal[i]))

        for i in range(self.psr[0].nTNECORR):
            name = 'ecorr_' + string(self.psr[0].TNECORRFlagVal[i])

            self.noisemodel[name] = noisepar(val=self.psr[0].TNECORRVal[i],
                                             flag=string(&self.psr[0].TNECORRFlagID[i][1]),
                                             flagval=string(self.psr[0].TNECORRFlagVal[i]))

        if self.psr[0].TNRedAmp != 0:
            self.noisemodel['log10_ared'] = self.psr[0].TNRedAmp

        if self.psr[0].TNRedGam != 0:
            self.noisemodel['gamma'] = self.psr[0].TNRedGam

        if self.psr[0].TNRedC != 0:
            self.noisemodel['nred'] = self.psr[0].TNRedC

        # set TOAs from input values
        self._inputtoas()

        if self.__input_toas is None:
            readTimfile(self.psr,timFile,self.npsr)           # load the arrival times (all pulsars)

    def _readpars(self,fixprefiterrors=True):
        cdef parameter *params = self.psr[0].param

        # create live proxies for all the parameters

        self.pardict = OrderedDict()

        for ct in range(MAX_PARAMS):
            for subct in range(params[ct].aSize):
                if fixprefiterrors and not params[ct].fitFlag[subct]:
                    params[ct].prefitErr[subct] = 0

                newpar = create_tempopar(params[ct],ct,subct,self.psr[0].eclCoord,self.units)
                newpar.err = params[ct].prefitErr[subct]
                self.pardict[newpar.name] = newpar

        for ct in range(1,self.psr[0].nJumps+1):  # jump 1 in the array not used...
            newpar = create_tempojump(&self.psr[0],ct,self.units)
            self.pardict[newpar.name] = newpar

        # the designmatrix plugin also adds extra parameters for sinusoidal whitening
        # but they don't seem to be used in the EPTA analysis
        # if(pPsr->param[param_wave_om].fitFlag[0]==1)
        #     nPol += pPsr->nWhite*2-1;

    # --- flags
    #     TO DO: proper flag interface
    #            flags() returns the list of defined flags
    #            flagvals(flagname,...) gets or sets arrays of values
    #            a possibility is also an obs interface
    def _readflags(self):
        cdef int i, j

        self.flagnames_ = []
        self.flags_ = dict()

        for i in range(self.nobs):
            for j in range(self.psr[0].obsn[i].nFlags):
                flag = string(self.psr[0].obsn[i].flagID[j][1:])

                if flag not in self.flagnames_:
                    self.flagnames_.append(flag)
                    # the maximum flag-value length is hard-set in tempo2.h
                    self.flags_[flag] = numpy.zeros(self.nobs,dtype=string_dtype + str(MAX_FLAG_LEN))

                self.flags_[flag][i] = string(self.psr[0].obsn[i].flagVal[j])

        for flag in self.flags_:
            self.flags_[flag].flags.writeable = False

    def _dimensionfy(self,array,unit):
        if self.units:
            return Quantity(array,unit=unit,copy=False)
        else:
            return array

    def _setstring(self,char* string,maxlen,value):
        value_bytes = value.encode()

        if len(value_bytes) < maxlen:
            stdio.sprintf(string,"%s",<char *>value_bytes)
        else:
            raise ValueError

    def _inputtoas(self):
        """Set up site arrival times based on input TOAs."""

        toas = self.__input_toas

        if toas is not None:
            if isinstance(toas, (list, numpy.ndarray, tuple, float, numpy.float128)):
                toamjd = numpy.atleast_1d(toas).astype(numpy.float128)
            else:
                # check if using an astropy time object
                try:
                    toamjd = numpy.atleast_1d(toas.mjd).astype(numpy.float128)  # make sure in MJD
                except Exception as e:
                    raise TypeError("Input TOAs are not of an allowed type: {}".format(e))

            self.psr[0].nobs = len(toamjd)
            self.nobs_ = len(toamjd)

            # set the values
            self._set_observation_from_input(toamjd)

    def _set_observation_from_input(self, toas):
        """Fill in all observation values from input TOAs."""

        cdef long double [:] _satday = <long double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].sat_day)
        cdef long double [:] _satsec = <long double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].sat_sec)
        cdef int [:] _deleted = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].deleted)
        cdef int [:] _nflags = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].nFlags)
        cdef double [:] _phaseoff = <double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].phaseOffset)
        cdef double [:] _dmerr = <double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].toaDMErr)
        cdef int [:] _obsNjump = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].obsNjump)
        cdef int [:] _obsNfdjump = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].obsNfdjump)
        cdef int [:] _jump = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].jump[0])
        cdef int [:] _fdjump = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].fdjump[0])
        cdef size_t obssize = sizeof(observation)
        _satday.strides[0] = obssize
        _satsec.strides[0] = obssize
        _deleted.strides[0] = obssize
        _nflags.strides[0] = obssize
        _phaseoff.strides[0] = obssize
        _dmerr.strides[0] = obssize
        _obsNjump.strides[0] = obssize
        _obsNfdjump.strides[0] = obssize
        _jump.strides[0] = obssize
        _fdjump.strides[0] = obssize

        # set the TOA values
        self.stoas[:] = toas

        # initialise other required values for each observation
        npsatday = numpy.asarray(_satday)
        npsatsec = numpy.asarray(_satsec)
        npdeleted = numpy.asarray(_deleted)
        npphaseoff = numpy.asarray(_phaseoff)
        npnflags = numpy.asarray(_nflags)
        npdmerr = numpy.asarray(_dmerr)
        npnjump = numpy.asarray(_obsNjump)
        npnfdjump = numpy.asarray(_obsNfdjump)
        npjump = numpy.asarray(_jump)
        npfdjump = numpy.asarray(_fdjump)

        days = numpy.floor(toas).astype(numpy.float128)
        npsatday[:] = days
        npsatsec[:] = toas - days

        npdeleted[:] = numpy.zeros(self.psr[0].nobs, dtype=numpy.int32)
        npphaseoff[:] = numpy.zeros(self.psr[0].nobs, dtype=numpy.float64)
        npnflags[:] = numpy.zeros(self.psr[0].nobs, dtype=numpy.int32)
        npdmerr[:] = numpy.zeros(self.psr[0].nobs, dtype=numpy.float64)
        npnjump[:] = numpy.ones(self.psr[0].nobs, dtype=numpy.int32)
        npnfdjump[:] = numpy.ones(self.psr[0].nobs, dtype=numpy.int32)
        npjump[:] = numpy.zeros(self.psr[0].nobs, dtype=numpy.int32)
        npfdjump[:] = numpy.zeros(self.psr[0].nobs, dtype=numpy.int32)

        # fill in fake filename
        for i in range(self.psr[0].nobs):
            # make sure fname array is empty
            memset(<char *>&(self.psr[0].obsn[i].fname[0]), 0, MAX_FILELEN * sizeof(char))
            strncpy(<char *>&(self.psr[0].obsn[i].fname[0]), "FAKE", 4 * sizeof(char))

        self._inputtoaerrs()
        self._inputobservatory()
        self._inputobsfreq()

    def _inputtoaerrs(self):
        """Set the TOA errors from input values."""
        
        cdef double [:] _toaerr = <double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].toaErr)
        cdef double [:] _origerr = <double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].origErr)
        cdef double [:] _efac = <double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].efac)
        cdef double [:] _equad = <double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].equad)
        cdef size_t obssize = sizeof(observation)
        _toaerr.strides[0] = obssize
        _origerr.strides[0] = obssize
        _efac.strides[0] = obssize
        _equad.strides[0] = obssize
        nptoaerr = numpy.asarray(_toaerr)
        nporigerr = numpy.asarray(_origerr)
        npefac = numpy.asarray(_efac)
        npequad = numpy.asarray(_equad)

        if self.__input_toas is not None:
            toaerrs = self.__input_toaerrs

            if toaerrs is None:
                raise ValueError("TOA errors must be supplied with input TOAs")

            if toaerrs is not None:
                if isinstance(toaerrs, (list, numpy.ndarray, tuple)):
                    if len(toaerrs) != self.psr[0].nobs:
                        raise ValueError("TOA errors must be same length as input TOAs")

                    toaerr = numpy.array(toaerrs, dtype=numpy.float64)
                elif isinstance(toaerrs, float):
                    # single value given
                    toaerr = toaerrs * numpy.ones(self.psr[0].nobs, dtype=numpy.float64)
                else:
                    raise TypeError("Input TOA errors are not of an allowed type")

            nptoaerr[:] = toaerr
            nporigerr[:] = numpy.copy(toaerr)  # store copy as original values

            # currently set EFAC and EQUAD to ones and zeros by default, respectively
            npefac[:] = numpy.ones(self.psr[0].nobs, dtype=numpy.float64)
            npequad[:] = numpy.zeros(self.psr[0].nobs, dtype=numpy.float64)

    def _inputobservatory(self):
        """Set the input TOA observatory site"""

        cdef int [:] _clockcorr = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].clockCorr)
        cdef int [:] _delaycorr = <int [:self.psr[0].nobs]>&(self.psr[0].obsn[0].delayCorr)
        cdef size_t obssize = sizeof(observation)
        _clockcorr.strides[0] = obssize
        _delaycorr.strides[0] = obssize

        npclockcorr = numpy.asarray(_clockcorr)
        npdelaycorr = numpy.asarray(_delaycorr)

        if self.__input_toas is not None:
            obs = self.__input_observatory
            if obs is None:
                raise ValueError("An observatory must be supplied with input TOAs")

            if isinstance(obs, (list, numpy.ndarray, tuple)):
                if len(obs) != self.psr[0].nobs:
                    raise ValueError("Number of supplied observatories must match TOAs")

                # observories will be truncated at 100 characters to fit in telID values
                obsv = numpy.array(obs, dtype="S100")
            elif isinstance(obs, str):
                # single value given
                obsv = numpy.array([obs for _ in range(self.psr[0].nobs)], dtype="S100")
            else:
                raise TypeError("Input TOA observatories are not of an allowed type")

            # set the observatories
            for i in range(self.psr[0].nobs):
                strncpy(<char *>&(self.psr[0].obsn[i].telID[0]), obsv[i], 100 * sizeof(char))

                # set which corrections to apply (taken from TEMPO2 readTimfile.C)
                if obsv[i][0] == "@" or obsv[i] == "bat":
                    # times are barycentric arrival times anyway, so don't correct
                    npclockcorr[i] = 0
                    npdelaycorr[i] = 0
                elif obsv[i] in ["STL", "STL_FBAT"]:
                    npclockcorr[i] = 0  # don't do clock corrections
                    npdelaycorr[i] = 1
                else:
                    npclockcorr[i] = 1
                    npdelaycorr[i] = 1

    def _inputobsfreq(self):
        """Set the input observation frequency (in MHz)"""

        cdef double [:] _freqsarr = <double [:self.psr[0].nobs]>&(self.psr[0].obsn[0].freq)
        _freqsarr.strides[0] = sizeof(observation)
        npfreqs = numpy.asarray(_freqsarr)

        if self.__input_toas is not None:
            freq = self.__input_obsfreq
            if freq is None:
                raise ValueError("Observing frequencies must be supplied with input TOAs")

            if isinstance(freq, (list, numpy.ndarray, tuple)):
                if len(freq) != self.psr[0].nobs:
                    raise ValueError("Number of supplied observation frequencies must match TOAs")

                freqs = numpy.array(freq, dtype=numpy.float64)
            elif isinstance(freq, (int, float)):
                # single value given
                freqs = freq * numpy.ones(self.psr[0].nobs, dtype=numpy.float64)
            else:
                raise TypeError("Input TOA observation frequencies are not of an allowed type")

            # set frequencies
            npfreqs[:] = freqs

    property name:
        """Get or set pulsar name."""

        def __get__(self):
            return string(self.psr[0].name)

        def __set__(self,value):
            # this is OK in both Python 2 and 3
            name_bytes = value.encode()

            if len(name_bytes) < 100:
                stdio.sprintf(self.psr[0].name,"%s",<char *>name_bytes)
            else:
                raise ValueError

    property binarymodel:
        """Get or set pulsar binary model."""

        def __get__(self):
            return string(self.psr[0].binaryModel)

        def __set__(self,value):
            model_bytes = value.encode()

            if len(model_bytes) < 100:
                stdio.sprintf(self.psr[0].binaryModel,"%s",<char *>model_bytes)
            else:
                raise ValueError

    property ephemeris:
        """Get or set the solar system ephemeris."""

        def __get__(self):
            return string(self.psr[0].ephemeris)

        def __set__(self,value):
            def seteph(filename,usecalceph=False):
                model_bytes = filename.encode()

                if len(model_bytes) < MAX_FILELEN:
                    stdio.sprintf(self.psr[0].JPL_EPHEMERIS,"%s",<char *>model_bytes)

                    # older tempo2 versions use ephemeris instead of JPL_EPHEMERIS for calceph.
                    stdio.sprintf(self.psr[0].ephemeris,    "%s",<char *>model_bytes)

                    self.psr[0].useCalceph = int(usecalceph)

                    if usecalceph and self.psr[0].t2cMethod == T2C_TEMPO:
                        print("Warning: calceph ephemerides are incompatible with T2CMETHOD=TEMPO. Reverting to IAU2000B.")
                        self.psr[0].t2cMethod = T2C_IAU2000B
                else:
                    raise ValueError("Ephemeris filename '{}' too long for tempo2.".format(filename))

            # by default, try the old-style version (e.g., DE430.1950.2000)
            if value == 'DE414':
                dates = '1960.2020'
            else:
                dates = '1950.2050'
            oldstyle = os.environ['TEMPO2'] + '/ephemeris/{0}.{1}'.format(value, dates)
            if os.path.isfile(oldstyle):
                seteph(oldstyle,False)
                return

            # next, look for a Spice file in the tempo2 directory
            newstyle = os.environ['TEMPO2'] + '/ephemeris/de{0}t.bsp'.format(value[2:5])
            if os.path.isfile(newstyle):
                seteph(newstyle,True)
                return

            # last, look in the current directory and assume Spice
            if os.path.isfile(value):
                seteph(value,True)
                return

            raise ValueError("Cannot find ephemeris file '{}'.".format(value))

    # TO DO: see if setting works
    property clock:
        """Get or set clock file."""

        def __get__(self):
            return string(self.psr[0].clock)

        def __set__(self,value):
            value_bytes = value.encode()

            if len(value_bytes) < 16:
                stdio.sprintf(self.psr[0].clock,"%s",<char *>value_bytes)
            else:
                raise ValueError("CLK name '{}' is too long.".format(value))


    excludepars = ['START','FINISH']

    # --- list parameters
    #     TODO: better way to exclude non-fit parameters
    def pars(self,which='fit'):
        """tempopulsar.pars(which='fit')

        Return tuple of parameter names:

        - if `which` is 'fit' (default), fitted parameters;
        - if `which` is 'set', all parameters with a defined value;
        - if `which` is 'all', all parameters."""

        if which == 'fit':
            return tuple(key for key in self.pardict if self.pardict[key].fit and key not in self.excludepars)
        elif which == 'set':
            return tuple(key for key in self.pardict if self.pardict[key].set)
        elif which == 'all':
            return tuple(self.pardict)
        elif isinstance(which,collections.Iterable):
            # to support vals() with which=sequence
            return which
        else:
            raise KeyError

    # --- number of observations
    property nobs:
        """Returns number of observations."""
        def __get__(self):
            return self.nobs_

    # --- number of fit parameters
    #     CHECK: inconsistent interface since ndim can change?
    property ndim:
        """Returns number of fit parameters."""
        def __get__(self):
            return sum(self.pardict[par].fit for par in self.pardict if par not in self.excludepars)

    # --- dictionary access to parameters
    #     TODO: possibly implement the full (nonmutable) dict interface by way of collections.Mapping
    def __contains__(self,key):
        return key in self.pardict

    def __getitem__(self,key):
        return self.pardict[key]

    # --- bulk access to parameter values
    def vals(self,values=None,which='fit'):
        """tempopulsar.vals(values=None,which='fit')

        Get (if no `values` provided) or set the parameter values, depending on `which`:

        - if `which` is 'fit' (default), fitted parameters;
        - if `which` is 'set', all parameters with a defined value;
        - if `which` is 'all', all parameters;
        - if `which` is a sequence, all parameters listed there.

        Parameter values are returned as a numpy longdouble array.

        Values to be set can be passed as a numpy array, sequence (in which case they
        are taken to correspond to parameters in the order given by `pars(which=which)`),
        or dict (in which case `which` will be ignored).

        Notes:

        - Passing values as anything else than numpy longdoubles may result in loss of precision.
        - Not all parameters in the selection need to be set.
        - Setting an unset parameter sets its `set` flag (obviously).
        - Unlike in earlier libstempo versions, setting a parameter does not set its error to zero."""

        if values is None:
            if self.units:
                return numpy.array([self.pardict[par].val for par in self.pars(which)],numpy.object)
            else:
                return numpy.fromiter((self.pardict[par].val for par in self.pars(which)),numpy.longdouble)
        elif isinstance(values,collections.Mapping):
            for par in values:
                self.pardict[par].val = values[par]
        elif isinstance(values,collections.Iterable):
            for par,val in zip(self.pars(which),values):
                self.pardict[par].val = val
        else:
            raise TypeError

    def errs(self,values=None,which='fit'):
        """tempopulsar.errs(values=None,which='fit')

        Same as `vals()`, but for parameter errors."""

        if values is None:
            if self.units:
                return numpy.array([self.pardict[par].err for par in self.pars(which)],numpy.object)
            else:
                return numpy.fromiter((self.pardict[par].err for par in self.pars(which)),numpy.longdouble)
        elif isinstance(values,collections.Mapping):
            for par in values:
                self.pardict[par].err = values[par]
        elif isinstance(values,collections.Iterable):
            for par,val in zip(self.pars(which),values):
                self.pardict[par].err = val
        else:
            raise TypeError

    def _timeify(self,array):
        if self.units:
            if self.clock[:2] == 'TT':
                timescale = 'tt'
            elif self.clock[:3] in ['TAI','UTC']:
                timescale = self.clock[:3].lower()
            else:
                raise NotImplementedError("Cannot recognize tempo2 CLK scale.")

            return Time(array,scale=timescale,format='mjd')
        else:
            return array

    def toas(self,updatebats=True):
        """tempopulsar.toas()

        Return computed SSB TOAs in MJD as a numpy.longdouble array.
        You get a copy of the current tempo2 array."""

        cdef long double [:] _toas = <long double [:self.nobs]>&(self.psr[0].obsn[0].bat)
        _toas.strides[0] = sizeof(observation)

        if updatebats:
            updateBatsAll(self.psr,self.npsr)

        return self._timeify(numpy.asarray(_toas).copy())

    # --- data access
    #     CHECK: proper way of doing docstring?
    property stoas:
        """Return site arrival times in MJD as a numpy.longdouble array.
        You get a view of the original tempo2 data structure, which you can write to."""

        def __get__(self):
            cdef long double [:] _stoas = <long double [:self.nobs]>&(self.psr[0].obsn[0].sat)
            _stoas.strides[0] = sizeof(observation)

            return numpy.asarray(_stoas)

    property roemer:
        """Return Roemer delay in seconds as a numpy.longdouble array."""

        def __get__(self):
            cdef long double [:] _roemer = <long double [:self.nobs]>&(self.psr[0].obsn[0].roemer)
            _roemer.strides[0] = sizeof(observation)

            return numpy.asarray(_roemer)

    property shapiro_sun:
        """Return Solar Shapiro delay in seconds as a numpy.double array."""

        def __get__(self):
            cdef double [:] _shapiro_sun = <double [:self.nobs]>&(self.psr[0].obsn[0].shapiroDelaySun)
            _shapiro_sun.strides[0] = sizeof(observation)

            return numpy.asarray(_shapiro_sun)

    property sun_ssb:
        def __get__(self):
            cdef double [:,:] _sun_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].sun_ssb[0])
            _sun_ssb.strides[0] = sizeof(observation)
            _sun_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_sun_ssb)

    # TO DO: need to dimensionfy as a Table
    property earth_ssb:
        def __get__(self):
            cdef double [:,:] _earth_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].earth_ssb[0])
            _earth_ssb.strides[0] = sizeof(observation)
            _earth_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_earth_ssb)

    property mercury_ssb:
        def __get__(self):
            cdef double [:,:] _mercury_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[0][0])
            _mercury_ssb.strides[0] = sizeof(observation)
            _mercury_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_mercury_ssb)

    property venus_ssb:
        def __get__(self):
            cdef double [:,:] _venus_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[1][0])
            _venus_ssb.strides[0] = sizeof(observation)
            _venus_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_venus_ssb)

    property mars_ssb:
        def __get__(self):
            cdef double [:,:] _mars_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[3][0])
            _mars_ssb.strides[0] = sizeof(observation)
            _mars_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_mars_ssb)

    property jupiter_ssb:
        def __get__(self):
            cdef double [:,:] _jupiter_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[4][0])
            _jupiter_ssb.strides[0] = sizeof(observation)
            _jupiter_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_jupiter_ssb)

    property saturn_ssb:
        def __get__(self):
            cdef double [:,:] _saturn_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[5][0])
            _saturn_ssb.strides[0] = sizeof(observation)
            _saturn_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_saturn_ssb)

    property uranus_ssb:
        def __get__(self):
            cdef double [:,:] _uranus_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[6][0])
            _uranus_ssb.strides[0] = sizeof(observation)
            _uranus_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_uranus_ssb)

    property neptune_ssb:
        def __get__(self):
            cdef double [:,:] _neptune_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[7][0])
            _neptune_ssb.strides[0] = sizeof(observation)
            _neptune_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_neptune_ssb)

    property pluto_ssb:
        def __get__(self):
            cdef double [:,:] _pluto_ssb = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].planet_ssb[8][0])
            _pluto_ssb.strides[0] = sizeof(observation)
            _pluto_ssb.strides[1] = sizeof(double)

            return numpy.asarray(_pluto_ssb)

    property observatory_earth:
        def __get__(self):
            cdef double [:,:] _observatory_earth = <double [:self.nobs,:6]>&(self.psr[0].obsn[0].observatory_earth[0])
            _observatory_earth.strides[0] = sizeof(observation)
            _observatory_earth.strides[1] = sizeof(double)

            return numpy.asarray(_observatory_earth)

    property psrPos:
        def __get__(self):
            cdef double [:,:] _psrPos = <double [:self.nobs,:3]>&(self.psr[0].obsn[0].psrPos[0])
            _psrPos.strides[0] = sizeof(observation)
            _psrPos.strides[1] = sizeof(double)

            return numpy.asarray(_psrPos)

    property toaerrs:
        """Returns TOA errors in units of microseconds as a numpy.double array.
        You get a view of the original tempo2 data structure, which you can write to."""

        def __get__(self):
            cdef double [:] _toaerrs = <double [:self.nobs]>&(self.psr[0].obsn[0].toaErr)
            _toaerrs.strides[0] = sizeof(observation)

            return self._dimensionfy(numpy.asarray(_toaerrs),u.us) if self.units else numpy.asarray(_toaerrs)

    # frequencies in MHz (numpy.double array)
    property freqs:
        """Returns observation frequencies in units of MHz as a numpy.double array.
        You get a view of the original tempo2 data structure, which you can write to."""

        def __get__(self):
            cdef double [:] _freqs = <double [:self.nobs]>&(self.psr[0].obsn[0].freq)
            _freqs.strides[0] = sizeof(observation)

            return self._dimensionfy(numpy.asarray(_freqs),u.MHz) if self.units else numpy.asarray(_freqs)

    property paramindex:
        def __get__(self):
            cdef int [:] _index = <int [:MAX_FIT]>&(self.psr[0].fitinfo.paramIndex[0])

            return numpy.asarray(_index)

    property paramcounters:
        def __get__(self):
            cdef int [:] _counters = <int [:MAX_FIT]>&(self.psr[0].fitinfo.paramCounters[0])

            return numpy.asarray(_counters)

    # --- Originally loaded value for SAT
    def origSats(self):
        """tempopulsar.origSats()

        Return originally loaded value of the SAT in case it is updated afterwards. Returned as long double array.
        You get a copy of the current values."""

        cdef long double [:] _origSats = <long double [:self.nobs]>&(self.psr[0].obsn[0].origsat)
        _origSats.strides[0] = sizeof(observation)

        return self._dimensionfy(numpy.asarray(_origSats),u.s) if self.units else numpy.asarray(_origSats)


	# --- day part of the SAT
    def satDay(self):
        """tempopulsar.satDay()

        Return the day part of the SAT as long double array.
        You get a copy of the current values."""

        cdef long double [:] _satDays = <long double [:self.nobs]>&(self.psr[0].obsn[0].sat_day)
        _satDays.strides[0] = sizeof(observation)

        return self._dimensionfy(numpy.asarray(_satDays),u.s) if self.units else numpy.asarray(_satDays)


    # --- return the decimal part of the SAT
    def satSec(self):
        """tempopulsar.satSec()

        Return decimal part of the SAT as a long double array
        You get a copy of the current values."""

        cdef long double [:] _satSecs  = <long double [:self.nobs]>&(self.psr[0].obsn[0].sat_sec)
        _satSecs.strides[0] = sizeof(observation)

        return self._dimensionfy(numpy.asarray(_satSecs),u.s) if self.units else numpy.asarray(_satSecs)


    # --- Correction to SSB
    def batCorrs(self):
        """tempopulsar.batCorrs()

        Return computed correction to SSB in units of days.
        You get a copy of the current values."""

        cdef long double [:] _batCorr = <long double [:self.nobs]>&(self.psr[0].obsn[0].batCorr)
        _batCorr.strides[0] = sizeof(observation)

        return self._dimensionfy(numpy.asarray(_batCorr),u.s) if self.units else numpy.asarray(_batCorr)

    # --- SSB frequencies
    #     CHECK: does updateBatsAll update the SSB frequencies?
    def ssbfreqs(self):
        """tempopulsar.ssbfreqs()

        Return computed SSB observation frequencies in units of MHz as a numpy.double array.
        You get a copy of the current values."""

        cdef double [:] _freqs = <double [:self.nobs]>&(self.psr[0].obsn[0].freqSSB)
        _freqs.strides[0] = sizeof(observation)

        updateBatsAll(self.psr,self.npsr)

        return self._dimensionfy(numpy.asarray(_freqs)/1e6,u.MHz) if self.units else numpy.asarray(_freqs)

    property deleted:
        """Return deletion status of individual observations (0 = OK, 1 = deleted)
        as a numpy.int array. You get a view of the original tempo2 data structure,
        which you can write to. Note that a numpy array of ints cannot be used directly
        to fancy-index another numpy array; for that, use `array[psr.deleted == 0]`
        or `array[~deletedmask()]`."""

        def __get__(self):
            cdef int [:] _deleted = <int [:self.nobs]>&(self.psr[0].obsn[0].deleted)
            _deleted.strides[0] = sizeof(observation)

            return numpy.asarray(_deleted)

    # --- deletion mask
    #     TO DO: support setting?
    def deletedmask(self):
        """tempopulsar.deletedmask()

        Returns a numpy.bool array of the delection station of observations.
        You get a copy of the current values."""

        return (self.deleted == 1)

    # --- flags
    def flags(self):
        """Returns the list of flags defined in this dataset (for at least some observations)."""

        return self.flagnames_

    # TO DO: setting flags
    def flagvals(self,flagname,values=None):
        """Returns (or sets, if `values` are given) a numpy unicode-string array
        containing the values of flag `flagname` for every observation."""

        if values is None:
            return self.flags_[flagname]
        else:
            raise NotImplementedError("Flag-setting capabilities are coming soon.")

    def pets(self,updatebats=True,formresiduals=True):
        """tempopulsar.pets()

        Return computed pulsar emission times in MJD as a numpy.longdouble array.
        You get a copy of the current tempo2 array."""

        cdef long double [:] _pets = <long double [:self.nobs]>&(self.psr[0].obsn[0].pet)
        _pets.strides[0] = sizeof(observation)

        if updatebats:
            updateBatsAll(self.psr,self.npsr)
        if formresiduals:
            formResiduals(self.psr,self.npsr,0)

        return self._timeify(numpy.asarray(_pets).copy())


    # --- residuals
    def residuals(self,updatebats=True,formresiduals=True,removemean=True, epoch=None, site=None, freq=None):
        """tempopulsar.residuals(updatebats=True,formresiduals=True,removemean=True)

        Returns residuals as a numpy.longdouble array (a copy of current values).
        Will update TOAs/recompute residuals if `updatebats`/`formresiduals` is True
        (default for both). Will remove residual mean if `removemean` is True;
        first residual if `removemean` is 'first'; weighted residual mean
        if `removemean` is 'weighted'.

        If `removemean` is `refphs` then the residuals will be referenced to the TZR
        parameters (TZRMJD, TZRSITE, TZRFREQ) given in the parameter file, or, if
        given, the `epoch`, `site` and `freq` values.
        """

        cdef long double [:] _res = <long double [:self.nobs]>&(self.psr[0].obsn[0].residual)
        _res.strides[0] = sizeof(observation)

        if removemean not in [True, False, 'weighted', 'first', 'refphs']:
            raise ValueError("Argument 'removemean' should be True, False, 'first', 'weighted', or 'refphs'.")

        if updatebats:
            updateBatsAll(self.psr,self.npsr)
        if formresiduals:
            if removemean != 'refphs':
                formResiduals(self.psr,self.npsr,1 if removemean is True else 0)
            else:
                # set an observation from which all residuals will be
                # referenced. Note this is equivalent to having a parameter
                # file containing the line REFPHS TZR and also having values
                # for TZRSITE, TZRMJD and TZRFREQ
                if epoch is None and site is None and freq is None and self.psr[0].refphs == REFPHS_TZR:
                    # refphs is already set
                    formResiduals(self.psr, self.npsr, 0)
                else:
                    # set values for the reference observation (stored within tzrsite, which gets
                    # updated within the TEMPO2 formBatsAll function)
                    if epoch is not None:
                        # set reference epoch
                        self["TZRMJD"].val = epoch
                    if site is not None:
                        strncpy(<char *>&(self.psr[0].tzrsite[0]), str.encode(site), 100 * sizeof(char))
                    if freq is not None:
                        self["TZRFRQ"] = freq
                    self.psr[0].refphs = REFPHS_TZR

                    # re-do BATS (to get BAT for reference epoch)
                    self.formbats()
                    formResiduals(self.psr, self.npsr, 0)

        res = numpy.asarray(_res).copy()
        if removemean is 'weighted':
            err = self.toaerrs
            res -= numpy.sum(res/err**2) / numpy.sum(1/err**2)
        elif removemean is 'first':
            # TO DO: what to do if there are deleted points?
            res -= res[0]

        return self._dimensionfy(res,u.s) if self.units else res

    def phaseresiduals(self, **kwargs):
        """
        Returns phase residuals (in cycles) as a numpy.longdouble array (a copy of
        current values). Arguments are the same as for the residuals method.
        """

        res = self.residuals(**kwargs)

        # convert to phase (in cycles)
        res *= self["F0"].val

        return self._dimensionfy(res, u.s / u.s) if self.units else res

    def formbats(self):
        formBatsAll(self.psr,self.npsr)

    def updatebats(self):
        updateBatsAll(self.psr,self.npsr)

    def formresiduals(self,removemean=True):
        formResiduals(self.psr,self.npsr,1 if removemean else 0)

    # TO DO: proper dimensionfy as a table
    def designmatrix(self,updatebats=True,fixunits=True,fixsigns=True,incoffset=True):
        """tempopulsar.designmatrix(updatebats=True,fixunits=True,incoffset=True)

        Returns the design matrix [nobs x (ndim+1)] as a numpy.longdouble array
        for current fit-parameter values. If fixunits=True, adjust the units
        of the design-matrix columns so that they match the tempo2
        parameter units. If fixsigns=True, adjust the sign of the columns
        corresponding to FX (F0, F1, ...) and JUMP parameters, so that
        they match finite-difference derivatives. If incoffset=False, the
        constant phaseoffset column is not included in the designmatrix."""

        # save the fit state of excluded pars
        excludeparstate = {}
        for par in self.excludepars:
            excludeparstate[par] = self[par].fit
            self[par].fit = False

        cdef FitInfo fitinfo

        fitinfo.paramCounters[0]   = 0
        fitinfo.paramDerivs[0]     = t2FitFunc_zero
        fitinfo.updateFunctions[0] = t2UpdateFunc_zero
        fitinfo.paramIndex[0]      = param_ZERO

        fitinfo.nParams = 1
        fitinfo.nConstraints = 0

        for par in self.pars():
            if self[par].isjump:
                fitinfo.paramIndex[fitinfo.nParams]      = param_JUMP
                fitinfo.paramCounters[fitinfo.nParams]   = self[par].subct
                fitinfo.paramDerivs[fitinfo.nParams]     = t2FitFunc_jump
                fitinfo.updateFunctions[fitinfo.nParams] = t2UpdateFunc_jump
                fitinfo.nParams = fitinfo.nParams + 1
            else:
                t2fit_fillOneParameterFitInfo(&self.psr[0],self[par].ct,self[par].subct,fitinfo)
                # the function already increases nParams

        try:
            assert fitinfo.nParams == self.ndim + 1
        except:
            print("Number of fitinfo parameters ({}) does not match fit parameters ({}).".format(fitinfo.nParams,self.ndim+1))
            raise

        cdef numpy.ndarray[double,ndim=2] ret = numpy.zeros((self.nobs,self.ndim+1),'d')

        cdef long double epoch = self.psr[0].param[param_pepoch].val[0]
        cdef observation *obsns = self.psr[0].obsn

        if updatebats:
            updateBatsAll(self.psr,self.npsr)

        cdef unsigned int idata, ipar
        for idata in range(self.nobs):
            for ipar in range(self.ndim + 1):
                ret[idata][ipar] = fitinfo.paramDerivs[ipar](&self.psr[0],0,obsns[idata].bbat - epoch,idata,fitinfo.paramIndex[ipar],fitinfo.paramCounters[ipar])

        if fixunits:
            dev, err = numpy.zeros(self.ndim + 1,'d'), numpy.ones(self.ndim + 1,'d')

            fp = self.pars()
            save = [self[p].err for p in fp]

            for ipar in range(self.ndim + 1):
                fitinfo.updateFunctions[ipar](&self.psr[0],0,fitinfo.paramIndex[ipar],fitinfo.paramCounters[ipar],dev[ipar],err[ipar])

            dev[0], dev[1:]  = 1.0, [self[p].err for p in fp]

            for p,v in zip(fp,save):
                self[p].err = v

            for i in range(self.ndim + 1):
                if dev[i] == 0:
                    print("Warning: design-matrix normalization coefficient is 0 for parameter {}. Proceeding without normalizing.")
                else:
                    ret[:,i] /= dev[i]

        if fixsigns:
            for i, par in enumerate(self.pars()):
                if (par[0] == 'F' and par[1] in '0123456789'):
                    ret[:,i+1] *= -1

        # restore the fit state of excluded pars
        for par in self.excludepars:
            self[par].fit = excludeparstate[par]

        return ret[:,0:] if incoffset else ret[:,1:]

    # --- observation telescope
    #     TO DO: support setting?
    def telescope(self):
        """tempopulsar.telescope()

        Returns a numpy character array of the telescope for each observation,
        mapping tempo2 `telID` values to names by way of the tempo2 runtime file
        `observatory/aliases`."""

        ret = numpy.zeros(self.nobs,dtype='a32')
        for i in range(self.nobs):
            ret[i] = string(self.psr[0].obsn[i].telID)
            if ret[i] in aliases:
                ret[i] = aliases[ret[i]]

        return ret

    # TOA filename
    def filename(self):
        """tempopulsar.filename()

        Returns a numpy character array of the filename for each observation,
        corresponding to tempo2 `fname` fields in the observation struct."""

        ret = numpy.zeros(self.nobs,dtype='a' + str(MAX_FILELEN))
        for i in range(self.nobs):
            ret[i] = string(self.psr[0].obsn[i].fname)

        return ret

    def binarydelay(self):
        """tempopulsar.binarydelay()

        Return a long-double numpy array of the delay introduced by the binary model.
        Does not reform residuals."""

        # TODO: Is it not much faster to call DDmodel/XXmodel directly?
        cdef long double [:] _torb = <long double [:self.nobs]>&(self.psr[0].obsn[0].torb)
        _torb.strides[0] = sizeof(observation)

        return numpy.asarray(_torb).copy()

    # TO DO: support setting?
    def elevation(self):
        """tempopulsar.elevation()

        Return a numpy double array of the elevation of the pulsar
        at the time of the observations."""

        cdef double [:] _posP = <double [:3]>self.psr[0].posPulsar
        cdef double [:] _zenith = <double [:3]>self.psr[0].obsn[0].zenith

        _posP.strides[0] = sizeof(double)
        posP = numpy.asarray(_posP)

        _zenith.strides[0] = sizeof(double)
        zenith = numpy.asarray(_zenith)

        elev = numpy.zeros(self.nobs)
        tels = self.telescope

        # TODO: make more Pythonic?
        for ii in range(self.nobs):
            obs = getObservatory(tels[ii])

            _zenith = <double [:3]>self.psr[0].obsn[ii].zenith
            zenith = numpy.asarray(_zenith)
            elev[ii] = numpy.arcsin(numpy.dot(zenith, posP) / obs.height_grs80) * 180.0 / numpy.pi

        return elev

    # --- phase jumps: this is Rutger's stuff, he should check the API
    # ---         RvH: Yep, I will, once I fully understand the tempo2
    # ---              implications. I am still adding/testing how to optimally
    # ---              use pulse numbers. See tempo2 formResiduals.C 1684--1700
    # ---              (I'll probably handle everything internally, and only
    # ---              rely on the pulse numbers given by tempo2)

    def phasejumps(self):
        """tempopulsar.phasejumps()

        Return an array of phase-jump tuples: (MJD, phase). These are
        copies.

        NOTE: As in tempo2, we refer the phase-jumps to site arrival times. The
        tempo2 definition of a phasejump is such that it is applied when
        observation_SAT > phasejump_SAT
        """
        npj = max(self.psr[0].nPhaseJump, 1)

        cdef int [:] _phaseJumpID = <int [:npj]>self.psr[0].phaseJumpID
        cdef int [:] _phaseJumpDir = <int [:npj]>self.psr[0].phaseJumpDir

        _phaseJumpID.strides[0] = sizeof(int)
        _phaseJumpDir.strides[0] = sizeof(int)

        phaseJumpID = numpy.asarray(_phaseJumpID)
        phaseJumpDir = numpy.asarray(_phaseJumpDir)

        phaseJumpMJD = self.stoas[phaseJumpID]

        npj = self.psr[0].nPhaseJump

        return numpy.column_stack((phaseJumpMJD[:npj], phaseJumpDir[:npj]))

    def add_phasejump(self, mjd, phasejump):
        """tempopulsar.add_phasejump(mjd,phasejump)

        Add a phase jump of value `phasejump` at time `mjd`.

        Note: due to the comparison observation_SAT > phasejump_SAT in tempo2,
        the exact MJD itself where the jump was added is not affected.
        """

        npj = self.psr[0].nPhaseJump

        # TODO: If we are at the maximum number of phase jumps, it should be
        # possible to remove a phase jump, or add to an existing one.
        # TODO: Do we remove the phase jump if it gets set to 0?
        if npj+1 > MAX_JUMPS:
            raise ValueError("Maximum number of phase jumps reached!")

        if self.nobs < 2:
            raise ValueError("Too few observations to allow phase jumps.")

        cdef int [:] _phaseJumpID = <int [:npj+1]>self.psr[0].phaseJumpID
        cdef int [:] _phaseJumpDir = <int [:npj+1]>self.psr[0].phaseJumpDir

        _phaseJumpID.strides[0] = sizeof(int)
        _phaseJumpDir.strides[0] = sizeof(int)

        phaseJumpID = numpy.asarray(_phaseJumpID)
        phaseJumpDir = numpy.asarray(_phaseJumpDir)

        if numpy.all(mjd < self.stoas) or numpy.all(mjd > self.stoas):
            raise ValueError("Cannot add a phase jump outside the dataset.")

        # Figure out to which observation we need to attach the phase jump
        fullind = numpy.arange(self.nobs)[self.stoas <= mjd]
        pjid = fullind[numpy.argmax(self.stoas[self.stoas <= mjd])]

        if pjid in phaseJumpID[:npj]:
            # This MJD already has a phasejump. Add to that jump
            jindex = numpy.where(phaseJumpID == pjid)[0][0]
            phaseJumpDir[jindex] += int(phasejump)
        else:
            # Add a new phase jump
            self.psr[0].nPhaseJump += 1

            phaseJumpID[-1] = pjid
            phaseJumpDir[-1] = int(phasejump)

    def remove_phasejumps(self):
        """tempopulsar.remove_phasejumps()

        Remove all phase jumps."""

        self.psr[0].nPhaseJump = 0

    property nphasejumps:
        """Return the number of phase jumps."""

        def __get__(self):
            return self.psr[0].nPhaseJump

    property pulse_number:
        """Return the pulse number relative to PEPOCH, as detected by tempo2

        WARNING: Will be deprecated in the future. Use `pulsenumbers`.
        """

        def __get__(self):
            cdef long long [:] _pulseN = <long long [:self.nobs]>&(self.psr[0].obsn[0].pulseN)
            _pulseN.strides[0] = sizeof(observation)

            return numpy.asarray(_pulseN)

    def pulsenumbers(self,updatebats=True,formresiduals=True,removemean=True):
        """Return the pulse number relative to PEPOCH, as detected by tempo2

        Returns the pulse numbers as a numpy array. Will update the
        TOAs/recompute residuals if `updatebats`/`formresiduals` is True
        (default for both). If that is requested, the residual mean is removed
        `removemean` is True. All this just like in `residuals`.
        """
        cdef long long [:] _pulseN = <long long [:self.nobs]>&(self.psr[0].obsn[0].pulseN)
        _pulseN.strides[0] = sizeof(observation)

        res = self.residuals(updatebats=updatebats, formresiduals=formresiduals,
                removemean=removemean)

        return numpy.asarray(_pulseN)

    def phase(self, **kwargs):
        """Return the pulse phase.

        Returns the pulse phase as a numpy array. Arguments are the same as for
        the `residuals` method.
        """

        cdef long double [:] _phase = <long double [:self.nobs]>&(self.psr[0].obsn[0].phase)
        _phase.strides[0] = sizeof(observation)

        _ = self.residuals(**kwargs)

        return numpy.asarray(_phase).copy()

    def _fit(self, renormalize=True, extrapartials=None, include_noise=True):
        # exclude deleted points
        mask = self.deleted == 0

        # limit points on either end if START and FINISH are marked as "fit"
        if self['START'].set and self['START'].fit:
            mask = mask & (self.stoas >= self['START'].val)

        if self['FINISH'].set and self['FINISH'].fit:
            mask = mask & (self.stoas <= self['FINISH'].val)

        res, err = self.residuals(removemean=False)[mask], self.toaerrs[mask]
        toas = numpy.double(self.stoas)[mask] * 86400
        M = self.designmatrix(updatebats=False,incoffset=True)[mask,:]

        # extra partials must be given as an nobs x nextra array, using the native obs ordering
        if extrapartials is not None:
            # may want to do more error checking here
            extrapar = extrapartials.shape[1]
            M = numpy.hstack((M, extrapartials[mask,:]))
        else:
            extrapar = 0

        err = err.copy()
        phiinv = numpy.zeros(M.shape[1])
        if self.noisemodel is not None and include_noise:
            for efac in [e for k,e in self.noisemodel.items() if k.startswith('efac')]:
                err[:] = numpy.where(self.flagvals(efac.flag)[mask] == efac.flagval,
                                     efac.val * err,err)

            for equad in [e for k,e in self.noisemodel.items() if k.startswith('equad')]:
                err[:] = numpy.where(self.flagvals(equad.flag)[mask] == equad.flagval,
                                     numpy.sqrt(err**2 + equad.val**2),err)

            # expand M matrix and create prior vector for extra noise
            if (self.noisemodel.get('log10_ared') is not None and
                    self.noisemodel.get('gamma') is not None):
                nred = self.noisemodel.get('nred', 100)
                F, Ffreqs = utils.create_fourier_design_matrix(toas, nred, freq=True)
                phi = utils.powerlaw(Ffreqs, log10_A=self.noisemodel['log10_ared'],
                                     gamma=self.noisemodel['gamma']) * Ffreqs[0]
                phiinv = numpy.concatenate((phiinv, 1/phi))
                M = numpy.hstack((M, F))

            if any(k.startswith('ecorr') for k in self.noisemodel):
                Umats = []
                for ecorr in [e for k,e in self.noisemodel.items() if k.startswith('ecorr')]:
                    flags = numpy.where(self.flagvals(ecorr.flag)[mask]==ecorr.flagval,
                                        self.flagvals(ecorr.flag), '')
                    _, U = utils.quantize_fast(toas, flags, dt=1.0)
                    Umats.append(U)
                    phi = numpy.ones(U.shape[1]) * (ecorr.val*1e-6)**2
                    phiinv = numpy.concatenate((phiinv, 1/phi))
                M = numpy.hstack((M, numpy.hstack(Umats)))

        # normalize the design matrix
        norm = numpy.sqrt(numpy.sum(M**2,axis=0))
        ntmpar = 1 + len(self.vals())
        if M.shape[1] > ntmpar:
            norm[ntmpar:] = 1
        if numpy.any(norm == 0):
            print("Warning: one or more of the design-matrix columns is null. Disabling renormalization (if active), but fit may fail.")
            renormalize = False

        if renormalize:
            M /= norm
        else:
            norm = numpy.ones_like(M[0,:])

        cinv = 1/(err * 1e-6)**2
        mtcm = numpy.dot(M.T, cinv[:,None]*M)
        mtcm += numpy.diag(phiinv)
        mtcy = numpy.dot(M.T, cinv*res)

        c = scipy.linalg.cho_factor(mtcm)
        xhat = scipy.linalg.cho_solve(c, mtcy)
        xvar = scipy.linalg.cho_solve(c, numpy.eye(len(mtcy)))

        # compute linearized chisq
        newres = res - numpy.dot(M, xhat)
        chisq = numpy.dot(newres, cinv*newres)

        # compute absolute estimates, normalized errors, covariance matrix
        ntmpar = 1 + len(self.vals())
        x = xhat/norm; x[1:ntmpar] += self.vals()
        err = numpy.sqrt(numpy.diag(xvar)) / norm
        cov = xvar / numpy.outer(norm,norm)

        # reset tempo2 parameter values
        self.vals(x[1:ntmpar])
        self.errs(err[1:ntmpar])

        return x, err, cov, chisq

    def fit(self,iters=1, renormalize=True, extrapartials=None, include_noise=True):
        """tempopulsar.fit(iters=1)

        Runs `iters` iterations of the a least-squares fit, using tempo2
        to compute the design matrix, and recomputing barycentric TOAs
        and residuals each time. Modifies parameter values and errors
        accordingly. Returns the tuple (xfit,stderr,covariance,chisq)
        after the last iteration. Note that these vectors and matrix
        are (ndim+1)- or (ndim+1)x(ndim+1)-dimensional, with the first
        row/column corresponding to a constant phase offset referenced
        to the first TOA (even if that point is not used)."""

        for i in range(iters):
            ret = self._fit(renormalize=renormalize,
                            extrapartials=extrapartials,
                            include_noise=include_noise)

        return ret

    # --- chisq
    def chisq(self,removemean='weighted'):
        """tempopulsar.chisq(removemean='weighted')

        Computes the chisq of current residuals vs errors,
        removing the noise-weighted residual, unless
        specified otherwise."""

        res, err = self.residuals(removemean=removemean), self.toaerrs

        return numpy.sum(res * res / (1e-12 * err * err))

    # --- rms residual
    def rms(self,removemean='weighted'):
        """tempopulsar.rms(removemean='weighted')

        Computes the current residual rms, removing the noise-weighted residual
        unless specified otherwise."""

        err = self.toaerrs
        norm = numpy.sum(1.0 / (1e-12 * err * err))

        res = math.sqrt(self.chisq(removemean=removemean)/norm)
        return self._dimensionfy(res,u.s) if self.units else res

    def savepar(self,parfile):
        """tempopulsar.savepar(parfile)

        Save current par file (calls tempo2's `textOutput(...)`)."""

        cdef char parFile[MAX_FILELEN]

        if not parfile:
            parfile = self.parfile

        parfile_bytes = parfile.encode()

        if len(parfile_bytes) > MAX_FILELEN - 1:
            raise IOError("Parfile name {0} too long for tempo2!".format(parfile))

        stdio.sprintf(parFile,"%s",<char *>parfile_bytes)

        # void textOutput(pulsar *psr,int npsr,
        #                 double globalParameter,  -- ?
        #                 int nGlobal,             -- ?
        #                 int outRes,              -- output residuals
        #                 int newpar, char *fname) -- write new par file
        textOutput(&(self.psr[0]),1,0,0,0,1,parFile)

        # tempo2/textOutput.C newer than revision 1.60 (2014/06/27 17:14:44) [~1.92 for tempo2.h]
        # does not honor parFile name, and uses pulsar_name + '-new.par' instead;
        # this was fixed in 1.61...
        # if tempo2version() >= StrictVersion("1.92"):
        #     os.rename(self.psr[0].name + '-new.par',parfile)

    def savetim(self,timfile):
        """tempopulsar.savetim(timfile)

        Save current par file (calls tempo2's `writeTim(...)`)."""

        cdef char timFile[MAX_FILELEN]

        if not timfile:
            timfile = self.timfile

        timfile_bytes = timfile.encode()

        if len(timfile_bytes) > MAX_FILELEN - 1:
            raise IOError("Timfile name {0} too long for tempo2!".format(timfile))

        stdio.sprintf(timFile,"%s",<char *>timfile_bytes)

        writeTim(timFile,&(self.psr[0]),'tempo2')


# access tempo2 utility function
def rad2hms(value):
    """rad2hms(value)

    Use tempo2 `turn_hms` to convert RAJ or DECJ to hours:minutes:seconds."""

    cdef char retstr[32]

    turn_hms(float(value/(2*math.pi)),<char *>&retstr)

    return retstr


def rewritetim(timfile):
    """rewritetim(timfile)

    Rewrite tim file to handle relative includes correctly. Not needed
    with tempo2 > 1.90."""

    import tempfile
    out = tempfile.NamedTemporaryFile(delete=False)

    for line in open(timfile,'r').readlines():
        if 'INCLUDE' in line:
            m = re.match('([ #]*INCLUDE) *(.*)',line)

            # encodes are needed here because file is open in binary mode
            if m:
                out.write('{0} {1}/{2}\n'.format(m.group(1),os.path.dirname(timfile),m.group(2)).encode())
            else:
                out.write(line.encode())
        else:
            out.write(line.encode())

    return out.name


def purgetim(timfile):
    """purgetim(timfile)

    Remove 'MODE 1' lines from tim file."""

    lines = filter(lambda l: 'MODE 1' not in l, open(timfile,'r').readlines())
    open(timfile,'w').writelines(lines)


# load observatory aliases from tempo2 runtime
aliases, ids = {}, {}
if 'TEMPO2' in os.environ:
    for line in open(os.environ['TEMPO2'] + '/observatory/aliases'):
        toks = line.split()

        if '#' not in line and len(toks) == 2:
            aliases[toks[1]] = toks[0]
            ids[toks[0]] = toks[1]
