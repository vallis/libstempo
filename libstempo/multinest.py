from __future__ import absolute_import, unicode_literals, print_function

import os, re, math
from ctypes import *
import numpy as N
from numpy.ctypeslib import as_array

# don't bother with parsing error
try:
    lib = cdll.LoadLibrary('libnest3.so')
except:
    lib = cdll.LoadLibrary(os.path.dirname(__file__) + '/libnest3.so')

# if we want to do OS X version detection:
# import platform
# if platform.system() == 'Darwin'
# '.'.join(platform.mac_ver().split('.')[:2]) --> 10.X

# libstempo.multinest.run borrows heavily from Johannes Buchner's pymultinest;
# it requires MultiNest v3.2 patched with cwrapper.f90

def run(LogLikelihood,
    Prior,
    n_dims,
    n_params = None,
    n_clustering_params = None, wrapped_params = None,
    importance_nested_sampling = True,
    multimodal = True, const_efficiency_mode = False, n_live_points = 400,
    evidence_tolerance = 0.5, sampling_efficiency = 0.8,
    n_iter_before_update = 100, null_log_evidence = -1e90,
    max_modes = 100, mode_tolerance = -1e90,
    outputfiles_basename = "./multinest-", seed = -1, verbose = False,
    resume = True, context = None, write_output = True, log_zero = -1e100,
    max_iter = 0, init_MPI = True, dump_callback = None):
    """
    Runs MultiNest

    The most important parameters are the two log-probability functions Prior
    and LogLikelihood. They are called by MultiNest.

    Prior should transform the unit cube into the parameter cube. Here
    is an example for a uniform prior::

        def Prior(cube, ndim, nparams):
            for i in range(ndim):
                cube[i] = cube[i] * 10 * math.pi

    The LogLikelihood function gets this parameter cube and should
    return the logarithm of the likelihood.
    Here is the example for the eggbox problem::

        def Loglike(cube, ndim, nparams):
            chi = 1.

            for i in range(ndim):
                chi *= math.cos(cube[i] / 2.)
            return math.pow(2. + chi, 5)

    Some of the parameters are explained below. Otherwise consult the
    MultiNest documentation.

    @param importance_nested_sampling:
        If True, Multinest will use Importance Nested Sampling (INS). Read http://arxiv.org/abs/1306.2144
        for more details on INS. Please read the MultiNest README file before using the INS in MultiNest v3.0.

    @param n_params:
        Total no. of parameters, should be equal to ndims in most cases
        but if you need to store some additional
        parameters with the actual parameters then you need to pass
        them through the likelihood routine.

    @param sampling_efficiency:
        defines the sampling efficiency. 0.8 and 0.3 are recommended
        for parameter estimation & evidence evalutation
        respectively.
        use 'parameter' or 'model' to select the respective default
        values

    @param mode_tolerance:
        MultiNest can find multiple modes & also specify which samples belong to which mode. It might be
        desirable to have separate samples & mode statistics for modes with local log-evidence value greater than a
        particular value in which case Ztol should be set to that value. If there isn't any particularly interesting
        Ztol value, then Ztol should be set to a very large negative number (e.g. -1e90).

    @param evidence_tolerance:
        A value of 0.5 should give good enough accuracy.

    @param n_clustering_params:
        If mmodal is T, MultiNest will attempt to separate out the
        modes. Mode separation is done through a clustering
        algorithm. Mode separation can be done on all the parameters
        (in which case nCdims should be set to ndims) & it
        can also be done on a subset of parameters (in which case
        nCdims < ndims) which might be advantageous as
        clustering is less accurate as the dimensionality increases.
        If nCdims < ndims then mode separation is done on
        the first nCdims parameters.

    @param null_log_evidence:
        If mmodal is T, MultiNest can find multiple modes & also specify
        which samples belong to which mode. It might be
        desirable to have separate samples & mode statistics for modes
        with local log-evidence value greater than a
        particular value in which case nullZ should be set to that
        value. If there isn't any particulrly interesting
        nullZ value, then nullZ should be set to a very large negative
        number (e.g. -1.d90).

    @param init_MPI:
        initialize MPI routines?, relevant only if compiling with MPI

    @param log_zero:
        points with loglike < logZero will be ignored by MultiNest

    @param max_iter:
        maximum number of iterations. 0 is unlimited.

    @param write_output:
        write output files? This is required for analysis.

    @param dump_callback:
        a callback function for dumping the current status

    """

    if n_params == None:
        n_params = n_dims
    if n_clustering_params == None:
        n_clustering_params = n_dims
    if wrapped_params == None:
        wrapped_params = [0] * n_dims

    WrappedType = c_int * len(wrapped_params)
    wraps = WrappedType(*wrapped_params)

    if sampling_efficiency == 'parameter':
        sampling_efficiency = 0.8
    if sampling_efficiency == 'model':
        sampling_efficiency = 0.3

    # MV 20130923

    loglike_type = CFUNCTYPE(c_double,
                             POINTER(c_double),c_int,c_int,c_void_p)

    dumper_type  = CFUNCTYPE(c_void_p,
                             c_int,c_int,c_int,
                             POINTER(c_double),POINTER(c_double),POINTER(c_double),
                             c_double,c_double,c_double,c_void_p)

    if hasattr(LogLikelihood,'loglike') and hasattr(Prior,'remap') and hasattr(Prior,'prior'):
        def loglike(cube,ndim,nparams,nullcontext):
            # we're not using context with libstempo.like objects

            pprior = Prior.premap(cube)

            # mappers are supposed to throw a ValueError if they get out of range
            try:
                pars = Prior.remap(cube)
            except ValueError:
                return -N.inf

            prior = pprior * Prior.prior(pars)
    
            return -N.inf if not prior else math.log(prior) + LogLikelihood.loglike(pars)
    else:
        def loglike(cube,ndim,nparams,nullcontext):
            # it's actually easier to use the context, if any, at the Python level
            # and pass a null pointer to MultiNest...

            args = [cube,ndim,nparams] + ([] if context is None else context)

            if Prior:
                Prior(*args)

            return LogLikelihood(*args)

    def dumper(nSamples,nlive,nPar,
               physLive,posterior,paramConstr,
               maxLogLike,logZ,logZerr,nullcontext):

        if dump_callback:
            # It's not clear to me what the desired PyMultiNest dumper callback
            # syntax is... but this should pass back the right numpy arrays,
            # without copies. Untested!
            pc =  as_array(paramConstr,shape=(nPar,4))

            dump_callback(nSamples,nlive,nPar,
                          as_array(physLive,shape=(nPar+1,nlive)).T,
                          as_array(posterior,shape=(nPar+2,nSamples)).T,
                          (pc[0,:],pc[1,:],pc[2,:],pc[3,:]),    # (mean,std,bestfit,map)
                          maxLogLike,logZ,logZerr)

    # MV 20130923: currently we support only multinest 3.2 (24 parameters),
    # but it would not be a problem to build up the parameter list dynamically

    lib.run(c_bool(importance_nested_sampling),c_bool(multimodal),c_bool(const_efficiency_mode),
            c_int(n_live_points),c_double(evidence_tolerance),
            c_double(sampling_efficiency),c_int(n_dims),c_int(n_params),
            c_int(n_clustering_params),c_int(max_modes),
            c_int(n_iter_before_update),c_double(mode_tolerance),
            create_string_buffer(outputfiles_basename.encode()),    # MV 20130923: need a regular C string
            c_int(seed),wraps,
            c_bool(verbose),c_bool(resume),
            c_bool(write_output),c_bool(init_MPI),
            c_double(log_zero),c_int(max_iter),
            loglike_type(loglike),dumper_type(dumper),
            c_void_p(0))

class multinestdata(dict):
    pass

class multinestpar(object):
    pass

# where are the multinest files?
def _findfiles(multinestrun,dirname,suffix='-post_equal_weights.dat'):
    # try chains/multinestrun-...
    #     chains/multinestrun/multinestrun-...
    root = [dirname + '/',dirname + '/' + multinestrun]
    
    # and if multinestrun is something like pulsar-model,
    # try chains/pulsar/model/pulsar-model-...
    if '-' in multinestrun:
        tokens = multinestrun.split('-')[:-1]
        pulsar, model = '-'.join(tokens[:-1]), tokens[-1]
        root.append(dirname + '/' + pulsar + '/' + model)

    return filter(lambda r: os.path.isfile(r + '/' + multinestrun + suffix),root)

def _getcomment(ret,filename):
    try:
        ret.comment = open(filename,'r').read()
    except IOError:
        pass

def _getmeta(ret,filename):
    try:
        meta = N.load(filename)
    except IOError:
        return

    ret.parnames  = list(meta['name'])
    ret.tempopars = list(meta['val'])   # somewhat legacy?
    ret.tempo = {}

    ml = N.argmax(ret.data[:,-1])

    for i,par in enumerate(ret.parnames):
        ret[par] = multinestpar()

        try:
            ret[par].val, ret[par].err = N.mean(ret.data[:,i]) + meta['offset'][i], math.sqrt(N.var(ret.data[:,i]))
            ret[par].offset = meta['offset'][i]
        except ValueError:
            ret[par].val, ret[par].err = N.mean(ret.data[:,i]), math.sqrt(N.var(ret.data[:,i]))

        if 'ml' in meta.dtype.names:
            ret[par].ml = meta['ml'][i]
        else:   
            ret[par].ml = ret.data[ml,i] + (meta['offset'][i] if 'offset' in meta.dtype.names else 0)

        ret.tempo[par] = multinestpar()
        ret.tempo[par].val, ret.tempo[par].err = meta['val'][i], meta['err'][i]

def load_mcmc(mcrun,dirname='.'):
    root = _findfiles(mcrun,dirname,'-chain.npy')

    ret = multinestdata()
    ret.dirname = root[0]

    alldata = N.load('{0}/{1}-chain.npy'.format(root[0],mcrun))

    # keep all the steps
    ret.data = alldata[:,:]

    _getmeta(ret,'{0}/{1}-meta.npy'.format(root[0],mcrun))
    _getcomment(ret,'{0}/{1}-comment.txt'.format(root[0],mcrun))

    return ret

def load_emcee(emceerun,dirname='.',chains=False):
    root = _findfiles(emceerun,dirname,'-chain.npy')

    ret = multinestdata()
    ret.dirname = root[0]

    alldata = N.load('{0}/{1}-chain.npy'.format(root[0],emceerun))

    # keep the last iteration of the walker cloud
    ret.data = alldata[:,-1,:]

    if chains:
        ret.chains = alldata

    _getmeta(ret,'{0}/{1}-meta.npy'.format(root[0],emceerun))
    _getcomment(ret,'{0}/{1}-comment.txt'.format(root[0],emceerun))

    return ret

def load(multinestrun,dirname='.'):
    root = _findfiles(multinestrun,dirname,'-post_equal_weights.dat')

    if not root:
        # try to find a tar.gz archive
        import tempfile, tarfile
        root = _findfiles(multinestrun,dirname,'.tar.gz')
        tar = tarfile.open('{0}/{1}.tar.gz'.format(root[0],multinestrun),mode='r|gz')
        root = [tempfile.mkdtemp(prefix='/tmp/')]
        tar.extractall(path=root[0])

    ret = multinestdata()
    ret.dirname = root[0]

    # get data
    ret.data = N.loadtxt('{0}/{1}-post_equal_weights.dat'.format(root[0],multinestrun))[:,:-1]

    # get evidence
    try:
        lines = open('{0}/{1}-stats.dat'.format(root[0],multinestrun),'r').readlines()
        try:
            ret.ev = float(re.search(r'Global Evidence:\s*(\S*)\s*\+/-\s*(\S*)',lines[0]).group(1))
        except:
            ret.ev = float(re.search(r'Global Log-Evidence           :\s*(\S*)\s*\+/-\s*(\S*)',lines[0]).group(1))
    except IOError:
        pass

    # get metadata
    _getmeta(ret,'{0}/{1}-meta.npy'.format(root[0],multinestrun))
    _getcomment(ret,'{0}/{1}-comment.txt'.format(root[0],multinestrun))

    if root[0][:4] == '/tmp':
        import shutil
        shutil.rmtree(root[0])

    return ret

def compress(rootname):
    import sys, os, glob

    dirname, filename = os.path.dirname(rootname), os.path.basename(rootname)

    if filename[-1] == '-':
        filename = filename[:-1]

    files = [filename + '-' + ending for ending in ('.txt','phys_live.points','stats.dat','ev.dat',
                                                    'post_equal_weights.dat','summary.txt','live.points',
                                                    'post_separate.dat','meta.npy','resume.dat','comment.txt')]

    cd = os.getcwd()
    os.chdir(dirname)

    os.system('tar zcf {0}.tar.gz {1}'.format(filename,' '.join(files)))

    files_exclude = [filename + '-' + ending for ending in ('IS.iterinfo','IS.points','IS.ptprob')]

    for f in files + files_exclude:
        if os.path.isfile(f):
            os.unlink(f)

    os.chdir(cd)
