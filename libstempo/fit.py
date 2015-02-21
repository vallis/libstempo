import numpy
import scipy.optimize

def chisq(psr):
    """Return the total chisq for the current timing solution,
    removing noise-averaged mean residual, and ignoring deleted points."""
    
    res, err = psr.residuals()[psr.deleted == 0], psr.toaerrs[psr.deleted == 0]
    res -= numpy.sum(res/err**2) / numpy.sum(1/err**2)
    cs = numpy.sum(res * res / (1e-12 * err * err))
    
    return cs

def dchisq(psr,deriv=False):
    """Return gradient of total chisq for the current timing solution,
    after removing noise-averaged mean residual, and ignoring deleted points."""
    
    res, err = psr.residuals()[psr.deleted == 0], psr.toaerrs[psr.deleted == 0]
    res -= numpy.sum(res/err**2) / numpy.sum(1/err**2)
    
    # bats already updated by residuals()
    M = psr.designmatrix(updatebats=False,fixunits=True,fixsigns=True)[psr.deleted==0,1:]

    # renormalize design-matrix columns
    norm = numpy.sqrt(numpy.sum(M**2,axis=0))
    M /= norm
        
    # compute chisq derivative, de-renormalize
    dr = -2 * numpy.dot(M.T,res / (1e-12 * err**2)) * norm
        
    return dr

def findmin(psr,method='Nelder-Mead',**kwargs):
    """Use scipy.optimize.minimize to find minimum-chisq timing solution,
    passing through all extra options. Resets psr[...].val to the final solution,
    and returns the final chisq. Will use chisq gradient if method requires it.
    Ignores deleted points."""

    ctr, err = psr.vals(), psr.errs()

    # to avoid losing precision, we're searching in units of parameter errors
    
    if numpy.any(err == 0.0):
        print("Warning: one or more fit parameters have zero a priori error, and won't be searched.")

    def func(xs):
        psr.vals([c + x*e for x,c,e in zip(xs,ctr,err)])
        return chisq(psr)
    
    def dfunc(xs):
        psr.vals([c + x*e for x,c,e in zip(xs,ctr,err)])
        return numpy.array([d*e for d,e in zip(dchisq(psr),err)])
    
    opts = kwargs.copy()
    if method not in ['Nelder-Mead','Powell']:
        opts['jac'] = dfunc

    res = scipy.optimize.minimize(func,[0.0]*len(ctr),method=method,**opts)
    print(res.message)
    
    # this will also set parameters to the minloc
    minchisq = func(res.x)
        
    return minchisq
