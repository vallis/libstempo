import os, math, re, time
from distutils.version import StrictVersion

import collections

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

# get zip-as-iterator behavior in Python 2
try:
    import itertools.izip as zip
except ImportError:
    pass

from libc cimport stdlib, stdio
from cython cimport view

import numpy
cimport numpy

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
    enum: MAX_FLAG_LEN
    enum: param_pepoch
    enum: param_raj
    enum: param_decj

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
        long double bat        # barycentric arrival time
        int deleted            # 1 if observation deleted, -1 if not in fit
        long double prefitResidual
        long double residual
        double toaErr          # error on TOA (in us)
        double toaDMErr        # error on TOA due to DM (in us)
        char **flagID          # ID of flags
        char **flagVal         # Value of flags
        int nFlags             # Number of flags set
        double freq            # frequency of observation (in MHz)
        double freqSSB         # frequency of observation in barycentric frame (in Hz)
        char telID[100]        # telescope ID
        double zenith[3]       # Zenith vector, in BC frame. Length=geodetic height
        long double torb       # Combined binary delay
        long long pulseN       # Pulse number

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
        int eclCoord            # = 1 for ecliptic coords otherwise celestial coords
        double posPulsar[3]     # 3-unitvector pointing at the pulsar
        #long double phaseJump[MAX_JUMPS] # Time of phase jump (Deprecated. WHY?)
        int phaseJumpID[MAX_JUMPS]        # ID of closest point to phase jump
        int phaseJumpDir[MAX_JUMPS]       # Size and direction of phase jump
        int nPhaseJump                    # Number of phase jumps
        double rmsPost

    void initialise(pulsar *psr, int noWarnings)
    void destroyOne(pulsar *psr)

    void readParfile(pulsar *psr,char parFile[][MAX_FILELEN],char timFile[][MAX_FILELEN],int npsr)
    void readTimfile(pulsar *psr,char timFile[][MAX_FILELEN],int npsr)

    void preProcess(pulsar *psr,int npsr,int argc,char *argv[])
    void formBatsAll(pulsar *psr,int npsr)
    void updateBatsAll(pulsar *psr,int npsr)                    # what's the difference?
    void formResiduals(pulsar *psr,int npsr,int removeMean)
    void doFit(pulsar *psr,int npsr,int writeModel)
    void updateParameters(pulsar *psr,int p,double val[],double error[])

    # for tempo2 versions older than ..., change to
    # void FITfuncs(double x,double afunc[],int ma,pulsar *psr,int ipos)
    void FITfuncs(double x,double afunc[],int ma,pulsar *psr,int ipos,int ipsr)

    int turn_hms(double turn,char *hms)

    void textOutput(pulsar *psr,int npsr,double globalParameter,int nGlobal,int outRes,int newpar,char *fname)
    void writeTim(char *timname,pulsar *psr,char *fileFormat)

    observatory *getObservatory(char *code)


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

    cdef int _isjump
    cdef void *_val
    cdef void *_err
    cdef int *_fitFlag
    cdef int *_paramSet

    def __init__(self,*args,**kwargs):
        raise TypeError("This class cannot be instantiated from Python.")

    property val:
        def __get__(self):
            if not self._isjump:
                return get_longdouble_as_scalar((<long double*>self._val)[0])
            else:
                return float((<double*>self._val)[0])

        def __set__(self,value):
            if not self._isjump:
                if not self._paramSet[0]:
                    self._paramSet[0] = 1

                set_longdouble(<long double*>self._val,value)
            else:
                (<double*>self._val)[0] = value
                # (<double*>self._err)[0] = 0

    property err:
        def __get__(self):
            if not self._isjump:
                return get_longdouble_as_scalar((<long double*>self._err)[0])
            else:
                return float((<double*>self._err)[0])

        def __set__(self,value):
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

    def __str__(self):
        if self.set:
            return 'tempo2 parameter %s (%s): %s +/- %s' % (self.name,'fitted' if self.fit else 'not fitted',repr(self.val),repr(self.err))
        else:
            return 'tempo2 parameter %s (unset)'

# since the __init__ for extension classes must have a Python signature,
# we use a factory function to initialize its attributes to pure-C objects

map_coords = {'RAJ': 'ELONG', 'DECJ': 'ELAT', 'PMRA': 'PMELONG', 'PMDEC': 'PMELAT'}

cdef create_tempopar(parameter par,int subct,int eclCoord):
    cdef tempopar newpar = tempopar.__new__(tempopar)

    newpar.name = par.shortlabel[subct].decode('ascii')

    if newpar.name in ['RAJ','DECJ','PMRA','PMDEC'] and eclCoord == 1:
        newpar.name = map_coords[newpar.name]

    newpar._isjump = 0

    newpar._val = &par.val[subct]
    newpar._err = &par.err[subct]
    newpar._fitFlag = &par.fitFlag[subct]
    newpar._paramSet = &par.paramSet[subct]

    return newpar

# TODO: note that currently we cannot change the number of jumps programmatically
cdef create_tempojump(pulsar *psr,int ct):
    cdef tempopar newpar = tempopar.__new__(tempopar)

    newpar.name = 'JUMP{0}'.format(ct)

    newpar._isjump = 1

    newpar._val = &psr.jumpVal[ct]
    newpar._err = &psr.jumpValErr[ct]
    newpar._fitFlag = &psr.fitJump[ct]

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

    def add_gwb(self,tempopulsar pulsar,distance=1):
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
        
        pulsar.stoas[:] += res[:] / 86400.0

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

# this is a Cython extension class; the benefit is that it can hold C attributes,
# but all attributes must be defined in the code

def tempo2version():
    return StrictVersion(str(TEMPO2_VERSION).split()[1])

cdef class tempopulsar:
    cpdef object parfile
    cpdef object timfile

    cdef int npsr           # number of pulsars
    cdef pulsar *psr        # array of pulsar structures

    cpdef object pardict    # dictionary of parameter proxies

    cpdef int nobs_         # number of observations

    cpdef object flagnames_ # a list of all flags that have values
    cpdef object flags_     # a dictionary of numpy arrays with flag values

    cpdef public double fitchisq    # chisq after tempo2 fit
    cpdef public double fitrms      # rms residuals after tempo2 fit

    # TO DO: is cpdef required here?
    cpdef jumpval, jumperr

    def __cinit__(self,parfile,timfile=None,warnings=False,fixprefiterrors=True,
                  dofit=False,maxobs=None):
        # initialize

        global MAX_PSR, MAX_OBSN

        self.npsr = 1

        # to save memory, only allocate space for this many pulsars and observations
        MAX_PSR = 1
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

        # preprocess the data

        preProcess(self.psr,self.npsr,0,NULL)
        formBatsAll(self.psr,self.npsr)

        # create parameter proxies

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

        if not os.path.isfile(timfile):
            # hail Mary pass
            maybe = '../tim/{0}'.format(timfile)
            if os.path.isfile(maybe):
                timfile = maybe
            else:
                raise IOError("Cannot find timfile {0}.".format(timfile))

        parfile_bytes, timfile_bytes = parfile.encode('ascii'), timfile.encode('ascii')

        for checkfile in [parfile_bytes,timfile_bytes]:
            if len(checkfile) > MAX_FILELEN - 1:
                raise IOError("Filename {0} is too long for tempo2.".format(checkfile))

        stdio.sprintf(parFile[0],"%s",<char *>parfile_bytes)
        stdio.sprintf(timFile[0],"%s",<char *>timfile_bytes)

        readParfile(self.psr,parFile,timFile,self.npsr)   # load the parameters    (all pulsars)
        readTimfile(self.psr,timFile,self.npsr)           # load the arrival times (all pulsars)

    def _readpars(self,fixprefiterrors=True):
        cdef parameter *params = self.psr[0].param

        # create live proxies for all the parameters

        self.pardict = OrderedDict()

        for ct in range(MAX_PARAMS):
            for subct in range(params[ct].aSize):
                if fixprefiterrors and not params[ct].fitFlag[subct]:
                    params[ct].prefitErr[subct] = 0

                newpar = create_tempopar(params[ct],subct,self.psr[0].eclCoord)
                newpar.err = params[ct].prefitErr[subct]
                self.pardict[newpar.name] = newpar

        for ct in range(1,self.psr[0].nJumps+1):  # jump 1 in the array not used...
            newpar = create_tempojump(&self.psr[0],ct)
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
                flag = self.psr[0].obsn[i].flagID[j][1:]
                flag = flag.decode('ascii')

                if flag not in self.flagnames_:
                    self.flagnames_.append(flag)
                    # the maximum flag-value length is hard-set in tempo2.h
                    self.flags_[flag] = numpy.zeros(self.nobs,dtype='U' + str(MAX_FLAG_LEN))

                flagvalue = self.psr[0].obsn[i].flagVal[j]
                self.flags_[flag][i] = flagvalue.decode('ascii')

        for flag in self.flags_:
            self.flags_[flag].flags.writeable = False

    property name:
        """Get or set pulsar name."""

        def __get__(self):
            return self.psr[0].name.decode('ascii')

        def __set__(self,value):
            name_bytes = value.encode('ascii')

            if len(name_bytes) < 100:
                stdio.sprintf(self.psr[0].name,"%s",<char *>name_bytes)
            else:
                raise ValueError

    property binarymodel:
        """Get or set pulsar binary model."""

        def __get__(self):
            return self.psr[0].binaryModel.decode('ascii')

        def __set__(self,value):
            model_bytes = value.encode('ascii')

            if len(model_bytes) < 100:    
                stdio.sprintf(self.psr[0].binaryModel,"%s",<char *>model_bytes)
            else:
                raise ValueError

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
        or dict (in which case which will be ignored).

        Notes:

        - Passing values as anything else than numpy longdoubles may result in loss of precision. 
        - Not all parameters in the selection need to be set.
        - Setting an unset parameter sets its `set` flag (obviously).
        - Unlike in earlier libstempo versions, setting a parameter does not set its error to zero."""
        
        if values is None:
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
            return numpy.fromiter((self.pardict[par].err for par in self.pars(which)),numpy.longdouble)
        elif isinstance(values,collections.Mapping):
            for par in values:
                self.pardict[par].err = values[par]
        elif isinstance(values,collections.Iterable):
            for par,val in zip(self.pars(which),values):
                self.pardict[par].err = val
        else:
            raise TypeError

    def toas(self,updatebats=True):
        """tempopulsar.toas()

        Return computed SSB TOAs in units of days as a numpy.longdouble array.
        You get a copy of the current tempo2 array."""

        cdef long double [:] _toas = <long double [:self.nobs]>&(self.psr[0].obsn[0].bat)
        _toas.strides[0] = sizeof(observation)

        if updatebats:
            updateBatsAll(self.psr,self.npsr)

        return numpy.asarray(_toas).copy()

    # --- data access
    #     CHECK: proper way of doing docstring?
    property stoas:
        """Return site arrival times in units of days as a numpy.longdouble array.
        You get a view of the original tempo2 data structure, which you can write to."""

        def __get__(self):
            cdef long double [:] _stoas = <long double [:self.nobs]>&(self.psr[0].obsn[0].sat)
            _stoas.strides[0] = sizeof(observation)

            return numpy.asarray(_stoas)

    property toaerrs:
        """Returns TOA errors in units of microseconds as a numpy.double array.
        You get a view of the original tempo2 data structure, which you can write to."""

        def __get__(self):
            cdef double [:] _toaerrs = <double [:self.nobs]>&(self.psr[0].obsn[0].toaErr)
            _toaerrs.strides[0] = sizeof(observation)

            return numpy.asarray(_toaerrs)

    # frequencies in MHz (numpy.double array)
    property freqs:
        """Returns observation frequencies in units of MHz as a numpy.double array.
        You get a view of the original tempo2 data structure, which you can write to."""

        def __get__(self):
            cdef double [:] _freqs = <double [:self.nobs]>&(self.psr[0].obsn[0].freq)
            _freqs.strides[0] = sizeof(observation)

            return numpy.asarray(_freqs)

    # --- SSB frequencies
    #     CHECK: does updateBatsAll update the SSB frequencies?
    def ssbfreqs(self):
        """tempopulsar.ssbfreqs()

        Return computed SSB observation frequencies in units of MHz as a numpy.double array.
        You get a copy of the current values."""

        cdef double [:] _freqs = <double [:self.nobs]>&(self.psr[0].obsn[0].freqSSB)
        _freqs.strides[0] = sizeof(observation)

        updateBatsAll(self.psr,self.npsr)

        return numpy.asarray(_freqs) / 1e6

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

    # --- residuals
    def residuals(self,updatebats=True,formresiduals=True,removemean=True):
        """tempopulsar.residuals(updatebats=True,formresiduals=True,removemean=True)

        Returns residuals as a numpy.longdouble array (a copy of current values).
        Will update TOAs/recompute residuals if `updatebats`/`formresiduals` is True
        (default for both). Will remove residual mean if `removemean` is True;
        will remove weighted residual mean if `removemean` is 'weighted'."""

        cdef long double [:] _res = <long double [:self.nobs]>&(self.psr[0].obsn[0].residual)
        _res.strides[0] = sizeof(observation)

        if removemean not in [True,False,'weighted']:
            raise ValueError("Argument 'removemean' should be True, False, or 'weighted'.")

        if updatebats:
            updateBatsAll(self.psr,self.npsr)
        if formresiduals:
            formResiduals(self.psr,self.npsr,1 if removemean is True else 0)

        res = numpy.asarray(_res).copy()
        if removemean is 'weighted':
            err = self.toaerrs
            res -= numpy.sum(res/err**2) / numpy.sum(1/err**2)

        return res

    def formbats(self):
        formBatsAll(self.psr,self.npsr)    

    def updatebats(self):
        updateBatsAll(self.psr,self.npsr)    

    def formresiduals(self,removemean=True):
        formResiduals(self.psr,self.npsr,1 if removemean else 0)    

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

        cdef int i
        cdef numpy.ndarray[double,ndim=2] ret = numpy.zeros((self.nobs,self.ndim+1),'d')

        cdef long double epoch = self.psr[0].param[param_pepoch].val[0]
        cdef observation *obsns = self.psr[0].obsn

        # the +1 is because tempo2 always fits for an arbitrary offset...
        cdef int ma = self.ndim + 1

        if updatebats:
            updateBatsAll(self.psr,self.npsr)

        for i in range(self.nobs):
            FITfuncs(obsns[i].bat - epoch,&ret[i,0],ma,&self.psr[0],i,0)

        cdef numpy.ndarray[double, ndim=1] dev, err

        if fixunits:
            dev, err = numpy.zeros(ma,'d'), numpy.ones(ma,'d')

            fp = self.pars()
            save = [self[p].err for p in fp]

            updateParameters(&self.psr[0],0,&dev[0],&err[0])
            dev[0], dev[1:]  = 1.0, [self[p].err for p in fp]

            for p,v in zip(fp,save):
                self[p].err = v

            for i in range(ma):
                ret[:,i] /= dev[i]

        if fixsigns:
            for i, par in enumerate(self.pars()):
                if (par[0] == 'F' and par[1] in '0123456789') or (par[:4] == 'JUMP'):
                    ret[:,i+1] *= -1

        # restore the fit state of excluded pars
        for par in self.excludepars:
            self[par].fit = excludeparstate[par]

        return ret[:,(0 if incoffset else 1):]

    # --- observation telescope
    #     TO DO: support setting?
    def telescope(self):
        """tempopulsar.telescope()

        Returns a numpy character array of the telescope for each observation,
        mapping tempo2 `telID` values to names by way of the tempo2 runtime file
        `observatory/aliases`."""

        ret = numpy.zeros(self.nobs,dtype='a32')
        for i in range(self.nobs):
            ret[i] = self.psr[0].obsn[i].telID
            if ret[i] in aliases:
                ret[i] = aliases[ret[i]]

        return ret

    def binarydelay(self):
        """tempopulsar.binarydelay()

        Return a long-double numpy array of the delay introduced by the binary model.
        Does not reform residuals."""

        # TODO: Is it not much faster to call DDmodel/XXmodel directly?
        cdef long double [:] _torb = <long double [:self.nobs]>&(self.psr[0].obsn[0].torb)
        _torb.strides[0] = sizeof(observation)

        return numpy.asarray(_torb).copy()

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

    # --- tempo2 fit
    #     CHECK: does mean removal affect the answer?
    def fit(self,iters=1):
        """tempopulsar.fit(iters=1)

        Runs `iters` iterations of the tempo2 fit, recomputing
        barycentric TOAs and residuals each time."""

        for i in range(iters):
            updateBatsAll(self.psr,self.npsr)
            formResiduals(self.psr,self.npsr,1)     # 1 to remove the mean

            doFit(self.psr,self.npsr,0)

        self.fitchisq = self.psr[0].fitChisq
        self.fitrms = self.psr[0].rmsPost

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

        Computes the current residual rms, 
        removing the noise-weighted residual, unless
        specified otherwise."""

        err = self.toaerrs
        norm = numpy.sum(1.0 / (1e-12 * err * err))

        return math.sqrt(self.chisq(removemean=removemean) / norm)

    def savepar(self,parfile):
        """tempopulsar.savepar(parfile)

        Save current par file (calls tempo2's `textOutput(...)`)."""

        cdef char parFile[MAX_FILELEN]

        if not parfile:
            parfile = self.parfile

        parfile_bytes = parfile.encode('ascii')

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

        timfile_bytes = timfile.encode('ascii')

        if len(timfile_bytes) > MAX_FILELEN - 1:
            raise IOError("Timfile name {0} too long for tempo2!".format(timfile))

        stdio.sprintf(timFile,"%s",<char *>timfile_bytes)

        writeTim(timFile,&(self.psr[0]),'tempo2');


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
            
            if m:
                out.write('{0} {1}/{2}\n'.format(m.group(1),os.path.dirname(timfile),m.group(2)).encode('ascii'))
            else:
                out.write(line.encode('ascii'))
        else:
            out.write(line.encode('ascii'))

    return out.name

def purgetim(timfile):
    """purgetim(timfile)

    Remove 'MODE 1' lines from tim file."""

    lines = filter(lambda l: 'MODE 1' not in l,open(timfile,'r').readlines())
    open(timfile,'w').writelines(lines)

# load observatory aliases from tempo2 runtime
aliases, ids = {}, {}
if 'TEMPO2' in os.environ:
    for line in open(os.environ['TEMPO2'] + '/observatory/aliases'):
        toks = line.split()

        if '#' not in line and len(toks) == 2:
            aliases[toks[1]] = toks[0]
            ids[toks[0]] = toks[1]
