from __future__ import print_function

import sys, math, types, re
import numpy as N, scipy.linalg as SL, scipy.special as SS

day = 24 * 3600
year = 365.25 * day

def dot(*args):
    return reduce(N.dot,args)

def _setuprednoise(pulsar,components=10):
    t = pulsar.toas()
    minx, maxx = N.min(t), N.max(t)
    x = (t - minx) / (maxx - minx)
    T = (day/year) * (maxx - minx)

    size = 2*components
    redF = N.zeros((pulsar.nobs,size),'d')
    redf = N.zeros(size,'d')

    for i in range(components):
        redF[:,2*i]   = N.cos(2*math.pi*(i+1)*x)
        redF[:,2*i+1] = N.sin(2*math.pi*(i+1)*x)

        redf[2*i] = redf[2*i+1] = (i+1) / T

    # include the normalization of the power-law prior in the Fourier matrices
    norm = year**2 / (12 * math.pi**2 * T)
    redF = math.sqrt(norm) * redF

    return redf, redF

def _quantize(times,dt=1):
    isort = N.argsort(times)
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    t = N.array([N.mean(times[l]) for l in bucket_ind],'d')
    
    U = N.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    return t, U

class Mask(object):
    def __init__(self,psr,usedeleted):
        self.usedeleted = usedeleted

        if self.usedeleted is False:
            self.deleted = psr.deleted

            if not N.any(self.deleted):
                self.usedeleted = True

    def __call__(self,array):
        if self.usedeleted is True:
            return array
        else:
            if array.ndim == 2:
                return array[~self.deleted,:]
            else:
                return array[~self.deleted]

def loglike(pulsar,efac=1.0,equad=None,jitter=None,Ared=None,gammared=None,marginalize=True,normalize=True,redcomponents=10,usedeleted=True):
    """Returns the Gaussian-process likelihood for 'pulsar'.

    The likelihood is evaluated at the current value of the pulsar parameters,
    as given by pulsar[parname].val.

    If efac, equad, and/or Ared are set, will compute the likelihood assuming
    the corresponding noise model. EFAC multiplies measurement noise;
    EQUAD adds in quadrature, and is given in us; red-noise is specified with
    the GW-like dimensionless amplitude Ared and exponent gamma, and is
    modeled with 'redcomponents' Fourier components.

    If marginalize=True (the default), loglike will marginalize over all the
    parameters in pulsar.fitpars, using an M-matrix formulation.
    """

    mask = Mask(pulsar,usedeleted)    

    err = 1.0e-6 * mask(pulsar.toaerrs)
    Cdiag = (efac*err)**2

    if equad:
        Cdiag = Cdiag + (1e-6*equad)**2 * N.ones(len(err))

    if Ared:
        redf, F = _setuprednoise(pulsar,redcomponents)
        F = mask(F)
        phi = Ared**2 * redf**(-gammared)

    if jitter:
        # quantize at 1 second; U plays the role of redF
        t, U = _quantize(86400.0 * mask(pulsar.toas()),1.0)
        phi_j = (1e-6*jitter)**2 * N.ones(U.shape[1])

        # stack the basis arrays if we're also doing red noise
        phi = N.hstack((phi,phi_j)) if Ared else phi_j
        F   = N.hstack((F,U))       if Ared else U

    if Ared or jitter:
        # Lentati formulation for correlated noise
        invphi = N.diag(1/phi)
        Ninv = N.diag(1/Cdiag)
        NinvF = dot(Ninv,F)          # could be accelerated
        X = invphi + dot(F.T,NinvF)  # invphi + FTNinvF

        Cinv = Ninv - dot(NinvF,N.linalg.inv(X),NinvF.T)
        logCdet = N.sum(N.log(Cdiag)) + N.sum(N.log(phi)) + N.linalg.slogdet(X)[1] # check
    else:
        # noise is all diagonal
        Cinv = N.diag(1/Cdiag)
        logCdet = N.sum(N.log(Cdiag))

    if marginalize:
        M = mask(pulsar.designmatrix())
        res = mask(N.array(pulsar.residuals(updatebats=False),'d'))
        
        CinvM = N.dot(Cinv,M)
        A = dot(M.T,CinvM)

        invA = N.linalg.inv(A)
        CinvMres = dot(res,CinvM)

        ret = -0.5 * dot(res,Cinv,res) + 0.5 * dot(CinvMres,invA,CinvMres.T) 

        if normalize:
            ret = ret - 0.5 * logCdet - 0.5 * N.linalg.slogdet(A)[1] - 0.5 * (M.shape[0] - M.shape[1]) * math.log(2.0*math.pi)
    else:
        res = mask(N.array(pulsar.residuals(),'d'))

        ret = -0.5 * dot(res,Cinv,res)

        if normalize:
            ret = ret - 0.5 * logCdet - 0.5 * len(res) * math.log(2.0*math.pi)

    return ret


standardpriors  = {'ECC':             (0,1),
                   'log10_efac':     (-1,1),'efac':     (0.1,10),
                   'log10_equad':    (-2,2),'equad':  (0.01,100),
                   'log10_jitter':   (-2,2),'jitter': (0.01,100),
                   'log10_Ared':  (-16,-10),'gammared':    (0,6)}

# [0,1] -> truncated positive normal
def map_posnormal(x0,sigma):
    erfc0 = 0.5 * SS.erfc(x0/(math.sqrt(2.0) * sigma))

    def map(x):
        if not (0 <= x <= 1.0): raise ValueError
        x = erfc0 + (1.0 - erfc0) * x
        return x0 - math.sqrt(2) * sigma * SS.erfinv(1.0 - 2.0*x)

    return map

# [0,1] -> normal distance with positive cut -> PX
def map_invposnormal(x0,sigma):
    erfc0 = 0.5 * SS.erfc(x0/(math.sqrt(2.0) * sigma))

    def map(x):
        if not (0 <= x <= 1.0): raise ValueError
        x = erfc0 + (1.0 - erfc0) * x
        r = 1.0/(x0 - math.sqrt(2) * sigma * SS.erfinv(1.0 - 2.0*x))
        if r < 0: raise ValueError
        return r

    return map

# correct sini sampling
# full mirror mapping is y = -1.0 + 2.0*x, return math.sqrt(1.0 - y**2)
def map_cosi2sini(sini_min,sini_max):
    y0, y1 = math.sqrt(1.0 - sini_max**2), math.sqrt(1.0 - sini_min**2)

    def map(x):
        if not (0 <= x <= 1.0): raise ValueError
        y = y0 + x * (y1 - y0)
        return math.sqrt(1.0 - y**2)

    return map

def map_cosi2sini_mirror():
    def map(x):
        if not (0 <= x <= 1.0): raise ValueError
        y = -1.0 + 2.0*x
        return math.sqrt(1.0 - y**2)

    return map

standardmaps    = {'SINI': map_cosi2sini(0,1)}

class tempopar(str):
    def __new__(cls,par):
        return str.__new__(cls,par)

    # maps [0,1] to parameter range
    def map(self,x):
        try:
            y0, y1 = self.range
            
            return y0 + x * (y1 - y0)
        except AttributeError:
            raise AttributeError('[ERROR] libstempo.like.tempopar.map: range is undefined for parameter {0}.'.format(self))

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self,val):
        self._range = val
        self.checkpriorvsrange()

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self,val):
        self._prior = val
        self.checkpriorvsrange()

    def checkpriorvsrange(self):
        # compare range and prior
        if (hasattr(self,'range') and hasattr(self,'prior') and isinstance(self.prior,(tuple,list))
                                  and (self.prior[0] > self.range[0] + getattr(self,'offset',0) or 
                                       self.prior[1] < self.range[1] + getattr(self,'offset',0)    ) ):
            print('[WARNING] libstempo.like.range: prior {0} is narrower than range {1}.'.format(self.prior,self.range))


def expandranges(parlist):
    """Rewrite a list of parameters by expanding ranges (e.g., log10_efac{1-10}) into
    individual parameters."""

    ret = []

    for par in parlist:
        # match anything of the form XXX{number1-number2}
        m = re.match('(.*)\{([0-9]+)\-([0-9]+)\}',par)

        if m is None:
            ret.append(par)
        else:
            # (these are strings)
            root, number1, number2 = m.group(1), m.group(2), m.group(3)

            # if number1 begins with 0s, number parameters as 00, 01, 02, ...,
            # otherwise go with 0, 1, 2, ...
            fmt = '{{0}}{{1:0{0}d}}'.format(len(number1)) if number1[0] == '0' else '{0}{1:d}'
            
            ret = ret + [fmt.format(root,i) for i in range(int(m.group(2)),int(m.group(3))+1)]

    return ret

# ordering of roots here is important
def _findrange(parlist,roots=['JUMP','DMXR1_','DMXR2_','DMX_','efac','log10_efac']):
    """Rewrite a list of parameters name by detecting ranges (e.g., JUMP1, JUMP2, ...)
    and compressing them."""

    rootdict = {root: [] for root in roots}

    res = []
    for par in parlist:
        found = False
        for root in roots:
            if len(par) > len(root) and par[:len(root)] == root:
                rootdict[root].append(int(par[len(root):]))
                found = True
        if not found:
            res.append(par)

    for root in roots:
        if rootdict[root]:
            if len(rootdict[root]) > 1:
                rmin, rmax = min(rootdict[root]), max(rootdict[root])
                res.append('{0}{{{1}-{2}}}{3}'.format(root,rmin,rmax,
                                                  '(incomplete)' if rmax - rmin != len(rootdict[root]) - 1 else ''))
            else:
                res.append('{0}{1}'.format(root,rootdict[root][0]))
    return res

class Prior(dict):
    def __init__(self,pulsar,parameters,prefit=False,rangemultiplier=4):
        self.searchpars = parameters
        self.fitpars = pulsar.fitpars
        self.setpars = [par for par in pulsar.setpars
                            if (par not in self.searchpars) and (par not in self.fitpars)]

        for par in parameters:
            self[par] = tempopar(par)

            if par in standardpriors:
                self[par].prior = standardpriors[par]
            
            if par in standardmaps:
                self[par].map = standardmaps[par]

            # if tempo2 has an error, use it to set the range
            if par in pulsar.setpars and pulsar[par].err != 0:
                if prefit:
                    val, err = pulsar.prefit[par].val, pulsar.prefit[par].err
                else:
                    val, err = pulsar[par].val, pulsar[par].err

                if abs(err/val) / sys.float_info.epsilon < 1000:
                    self[par].offset = val
                    val = 0.0

                self[par].range = (val - rangemultiplier*err,val + rangemultiplier*err)
            elif par in standardpriors:
                self[par].range = standardpriors[par]

    @property
    def meta(self):
        def partuple(par):
            return (getattr(self[par],'offset',0),)

        return N.fromiter((partuple(par) for par in self.searchpars),
                          dtype=[('offset','f16')])

    def report(self):
        print()
        print("==== libstempo.like.Prior report ====")
        print("  Search        parameters: {0}".format(' '.join(_findrange(self.searchpars))))
        print("  Marginalized  parameters: {0}".format(' '.join(_findrange(self.fitpars))))
        print("  Other set     parameters: {0}".format(' '.join(_findrange(self.setpars))))

        ranges = []
        for par in self.searchpars:
            if isinstance(self[par].map,types.FunctionType):
                ranges.append('<map>')
            elif hasattr(self[par],'range'):
                ranges.append(str(self[par].range))
            else:
                ranges.append('')

        priors = []
        for par in self.searchpars:
            if hasattr(self[par],'prior') and isinstance(self[par].prior,types.FunctionType):
                priors.append('<map>')
            elif hasattr(self[par],'prior'):
                priors.append(str(self[par].prior))
            else:
                priors.append('')            

        offsets = []
        for par in self.searchpars:
            if hasattr(self[par],'offset'):
                offsets.append(repr(self[par].offset))
            else:
                offsets.append('')

        # may want to consider also titles here
        # eventually we'll make this into a separate function to handle avgs + vars, etc.
        lens = [max(6,max(map(len,l))) for l in [self.searchpars,ranges,priors,offsets]]

        print("  Search ranges and priors:")
        line = '    {{0:{0}s}} | {{1:{1}s}} | {{2:{2}s}} | {{3:{3}s}}'.format(*lens)
        print(line.format('PAR','RANGE','PRIOR','OFFSET'))
        for p in zip(self.searchpars,ranges,priors,offsets):
            print(line.format(*p))

        print()

    # as is, this is a "cube" prior suitable for multinest integration
    # it takes transformed parameters, so offsets don't matter
    def prior(self,pardict):
        # reconstruct a dictionary if we're given a sequence 
        if not isinstance(pardict,dict):
            pardict = {par: pardict[i] for (i,par) in enumerate(self.searchpars)}

        prior = 1.0

        for par in self.searchpars:
            # prior is implicitly one for parameters that don't have one
            if not hasattr(self[par],'prior'):
                continue

            parP = self[par].prior

            if hasattr(parP,'__call__'):
                # prior is a function
                # currently allows only separable priors since it passes a single argument
                prior = prior * parP(pardict[par])
            elif isinstance(parP,(tuple,list)):
                # prior is an interval
                if parP[0] <= pardict[par] <= parP[1]:
                    # handle priors with (semi-)infinite prior intervals
                    if hasattr(self[par],'range') and (parP[0] != -N.inf) and (parP[1] != N.inf):
                        # compute correct Ockham penalty even if we're restricting the range
                        prior = prior * float((self[par].range[1] - self[par].range[0]) / (parP[1] - parP[0]))
                    # implicit else: prior = prior * 1
                else:
                    return 0
            else:
                # prior is a number
                prior = prior * parP

            # shortcircuit null prior
            if not prior:
                return 0

        return prior

    # remap point in [0,1]^n cube to dictionary of search parameters, using individual map functions
    # (also changes xs, as required by multinest, but does not include offsets there)
    def remap(self,xs):
        pardict = {}

        for i,par in enumerate(self.searchpars):
            xs[i] = self[par].map(xs[i])
            pardict[par] = xs[i] + getattr(self[par],'offset',0)

        return pardict

    def remap_list(self,xs):
        return [self[par].map(xs[i]) for i,par in enumerate(self.searchpars)]

    def premap(self,xs):
        pprior = 1.0

        for i,par in enumerate(self.searchpars):
            if hasattr(self[par],'preprior'):
                pprior = pprior * self[par].preprior(xs[i])

            if hasattr(self[par],'premap'):
                xs[i] = self[par].premap(xs[i])

        return pprior

# still incomplete, but the idea is to show the figures that differ between str1 and str2 in bold
def _showdiff(str1,str2):
    if '(' in str1 and '(' in str2:
        for i in range(min(len(str1),len(str2))):
            if str1[i] == '(' or str2[i] == '(' or str1[i] != str2[i]:
                break

        return ('\033[1m{0}\033[0m{1}'.format(str1[:i],str1[i:]),
                '\033[1m{0}\033[0m{1}'.format(str2[:i],str2[i:]))
    else:
        return str1, str2

def _formatval(val,err,showerr=True):
    if N.isnan(val):
        # no value
        return 'nan'
    elif val == 0:
        # just zero
        if err == 0 or N.isnan(err):
            return '0'
        else:
            return '0 +/- %.1e' % err
    elif err == 0 or N.isnan(err):
        # no error
        return '%+.15e' % val
    elif abs(val) < err:
        # error larger than value (but value is not zero)
        if showerr:
            return '%+.1e +/- %.1e' % (val,err)
        else:
            return '%+.1e' % val

    # general case: set the precision of the value at the magnitude of the error
    prec = int(math.floor(math.log10(abs(val))) - math.floor(math.log10(abs(err))) + 1)
    
    # format the value (can't use format() for N.longdouble), then grab the mantissa and exponent
    str1 = '%+.*e' % (prec,val)
    mantissa, exponent = str1.split('e')
    exponent = int(exponent)

    # interject the error, with two digits of precision, if requested    
    # need to do business with sign because format() does not honor '+' and zero-padding together
    if showerr:
        errstr = int(abs(err) / (10**(exponent - prec)) + 0.5)
        return '{0}({1})e{2}{3:02d}'.format(mantissa,errstr,'-' if exponent < 0 else '+',abs(exponent))
    else:
        return '{0}e{1}{2:02d}'.format(mantissa,'-' if exponent < 0 else '+',abs(exponent))        


import contextlib

# should be in libstempo.util?
@contextlib.contextmanager
def numpy_seterr(**kwargs):
    old_settings = N.seterr(**kwargs)

    try:
        yield
    finally:
        N.seterr(**old_settings)


class Loglike(object):
    def __init__(self,pulsar,parameters,redcomponents=10):
        self.psr, self.searchpars = pulsar, parameters
        self.multiefac = sum('efac' in par for par in parameters) > 1

        self.pars = []
        self.efac, self.equad, self.Ared, self.jitter = None, None, None, None

        for par in parameters:
            if ('efac' in par) or ('equad' in par) or ('Ared' in par) or ('jitter' in par) or (par == 'gammared'):
                if par[:6] == 'log10_':
                    setattr(self,par[6:],lambda d, parname=par: 10.0**d[parname])
                else:
                    setattr(self,par,lambda d, parname=par: d[parname])
            else:
                # these are the "real" tempo2 parameters
                if par not in self.psr:
                    raise KeyError("[ERROR] libstempo.like.Loglike: parameter {0} unknown.".format(par))
                elif self.psr[par].fit == True:
                    raise ValueError("[ERROR] libstempo.like.Loglike: trying to set marginalized parameter {0}.".format(par))
                else:
                    self.pars.append(par)

        if self.multiefac:
            self.sysflags = list(set(self.psr.flagvals('sys')))
            self.sysflags.sort()

            longest = str(len(self.sysflags)-1)
            self.efacpars = ['efac{0:0{1}d}'.format(i,len(longest)) for i in range(len(self.sysflags))]

            self.err2 = [(1.0e-6 * (self.psr.flagvals('sys') == sys))**2 for sys in self.sysflags]

            if N.any([not hasattr(self,efacpar) for efacpar in self.efacpars]):
                raise KeyError("[ERROR] libstempo.like.Loglike: when multiefac=True, you need to fit (log){0}--{1}.".format(self.efacpars[0],self.efacpars[-1]))
        else:
            self.err2 = (1.0e-6 * self.psr.toaerrs)**2

        if self.equad:
            self.ones = N.ones(len(self.psr.toaerrs))

        if self.Ared:
            self.redf, self.redF = _setuprednoise(self.psr,redcomponents)

        if self.jitter:
            # quantize at 1 second; U plays the role of redF
            t, U = _quantize(86400.0 * pulsar.toas(),1.0)

            self.ones2 = N.ones(U.shape[1]) # should it be self.twos? :)
            # stack the basis arrays if we're also doing red noise
            self.redF = N.hstack((self.redF,U)) if self.Ared else U

        self.marginalize = len(self.psr.fitpars) > 0

        if not self.pars:
            self.M = self.psr.designmatrix()
            self.res = N.array(self.psr.residuals(updatebats=False),'d')

        def partuple(par):
            if par in pulsar.setpars:
                return (par,pulsar[par].val,pulsar[par].err,pulsar.prefit[par].val,pulsar.prefit[par].err)
            else:
                return (par,N.nan,N.nan,N.nan,N.nan)

        self.meta = N.fromiter((partuple(par) for par in parameters),
                               dtype=[('name','a32'),('val','f16'),('err','f16'),('pval','f16'),('perr','f16')])

    def __call__(self,pardict):
        return self.loglike(pardict)

    def loglike(self,pardict):
        # reconstruct a dictionary if we're given a sequence 
        if not isinstance(pardict,dict):
            pardict = {par: pardict[i] for (i,par) in enumerate(self.searchpars)}

        for par in self.pars:
            self.psr[par].val = pardict[par]

        if self.multiefac:
            Cdiag = sum(getattr(self,self.efacpars[i])(pardict)**2 * self.err2[i] for i in range(len(self.sysflags)))
        else:
            if self.efac:
                Cdiag = self.efac(pardict)**2 * self.err2
            else:
                Cdiag = self.err2

        if self.equad:
            Cdiag = Cdiag + (1e-6*self.equad(pardict))**2 * self.ones

        if self.Ared:
            invphi = self.Ared(pardict)**-2 * self.redf**self.gammared(pardict)

        if self.jitter:
            invphi_j = (1e-6*self.jitter(pardict))**-2 * self.ones2
            invphi = N.hstack((invphi,invphi_j)) if self.Ared else invphi_j

        if self.Ared or self.jitter:
            Ninv = N.diag(1/Cdiag)
            NinvF = dot(Ninv,self.redF)
            X = N.diag(invphi) + dot(self.redF.T,NinvF)

            Cinv = Ninv - dot(NinvF,N.linalg.inv(X),NinvF.T)
            logCdet = N.sum(N.log(Cdiag)) - N.sum(N.log(invphi)) + N.linalg.slogdet(X)[1] # check
        else:
            Cinv = N.diag(1/Cdiag)
            logCdet = N.sum(N.log(Cdiag))

        if self.marginalize:
            if not self.pars:
                # tempo2 parameters don't change, so use cached
                M, res = self.M, self.res
            else:
                M = self.psr.designmatrix()
                res = N.array(self.psr.residuals(updatebats=False),'d')

            CinvM = N.dot(Cinv,M)
            A = dot(M.T,CinvM)

            invA = N.linalg.inv(A)
            CinvMres = dot(res,CinvM)

            # TO DO: should check that we don't need factors of 2*pi for the F formalism
            loglike = (- 0.5 * dot(res,Cinv,res) + 0.5 * dot(CinvMres,invA,CinvMres.T) - 0.5 * logCdet - 0.5 * N.linalg.slogdet(A)[1]
                       - 0.5 * (M.shape[0] - M.shape[1]) * math.log(2.0*math.pi))
        else:
            res = N.array(self.psr.residuals(),'d')

            loglike = (- 0.5 * dot(res,Cinv,res) - 0.5 * logCdet - 0.5 * len(res) * math.log(2.0*math.pi))

        return loglike

    # currently not reporting on MCMC ML
    # also report ML and RMS at the end of the parameters
    def report(self,data):
        tempo = [_formatval(data.tempo[par].val,data.tempo[par].err) for par in self.searchpars]
        mcmc  = [_formatval(data[par].val,      data[par].err)       for par in self.searchpars]

        delta  = [_formatval(data[par].val-data.tempo[par].val,max(data[par].err,data.tempo[par].err),showerr=False) for par in self.searchpars]

        with numpy_seterr(divide='ignore'):
            ratio  = [('%.1e'  % (data[par].err/data.tempo[par].err)) for par in self.searchpars]
            fdelta = [('%+.1e' % ((data[par].val-data.tempo[par].val)/data.tempo[par].err)) for par in self.searchpars]

        print()
        print("==== libstempo.like.Loglike report ====")

        lens = [max(6,max(map(len,l))) for l in [self.searchpars,tempo,mcmc,delta,ratio,fdelta]]

        print("  Tempo2 ML values and MCMC conditional means:")
        line = '    {{0:{0}s}} | {{1:{1}s}} | {{2:{2}s}} | {{3:{3}s}} | {{4:{4}s}} | {{5:{5}s}}'.format(*lens)
        print(line.format('PAR','TEMPO2','MCMC','DIFF','ERAT','BIAS'))
        for p in zip(self.searchpars,tempo,mcmc,delta,ratio,fdelta):
            print(line.format(*p))

        print()
