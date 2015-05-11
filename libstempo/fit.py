import numpy
import scipy.linalg, scipy.optimize

def chisq(psr,formbats=False):
    """Return the total chisq for the current timing solution,
    removing noise-averaged mean residual, and ignoring deleted points."""
    
    if formbats:
        psr.formbats()

    res, err = psr.residuals(removemean=False)[psr.deleted == 0], psr.toaerrs[psr.deleted == 0]
    
    res -= numpy.sum(res/err**2) / numpy.sum(1/err**2)

    return numpy.sum(res * res / (1e-12 * err * err))

def dchisq(psr,formbats=False,renormalize=True):
    """Return gradient of total chisq for the current timing solution,
    after removing noise-averaged mean residual, and ignoring deleted points."""
    
    if formbats:
        psr.formbats()

    res, err = psr.residuals(removemean=False)[psr.deleted == 0], psr.toaerrs[psr.deleted == 0]

    res -= numpy.sum(res/err**2) / numpy.sum(1/err**2)
    
    # bats already updated by residuals(); skip constant-phase column
    M = psr.designmatrix(updatebats=False,fixunits=True,fixsigns=True)[psr.deleted==0,1:]

    # renormalize design-matrix columns
    if renormalize:
        norm = numpy.sqrt(numpy.sum(M**2,axis=0))
        M /= norm
    else:
        norm = 1.0

    # compute chisq derivative, de-renormalize
    dr = -2 * numpy.dot(M.T,res / (1e-12 * err**2)) * norm
    
    return dr

def findmin(psr,method='Nelder-Mead',history=False,formbats=False,renormalize=True,bounds={},**kwargs):
    """Use scipy.optimize.minimize to find minimum-chisq timing solution,
    passing through all extra options. Resets psr[...].val to the final solution,
    and returns the final chisq. Will use chisq gradient if method requires it.
    Ignores deleted points."""

    ctr, err = psr.vals(), psr.errs()

    # to avoid losing precision, we're searching in units of parameter errors
    
    if numpy.any(err == 0.0):
        print("Warning: one or more fit parameters have zero a priori error, and won't be searched.")

    hloc, hval = [], []

    def func(xs):
        psr.vals([c + x*e for x,c,e in zip(xs,ctr,err)])

        ret = chisq(psr,formbats=formbats)

        if numpy.isnan(ret):
            print("Warning: chisq is nan at {0}.".format(psr.vals()))

        if history:
            hloc.append(psr.vals())
            hval.append(ret)

        return ret

    def dfunc(xs):
        psr.vals([c + x*e for x,c,e in zip(xs,ctr,err)])

        dc = dchisq(psr,formbats=formbats,renormalize=renormalize)
        ret = numpy.array([d*e for d,e in zip(dc,err)],'d')

        return ret

    opts = kwargs.copy()

    if method not in ['Nelder-Mead','Powell']:
        opts['jac'] = dfunc

    if method in ['L-BFGS-B']:
        opts['bounds'] = [(float((bounds[par][0] - ctr[i])/err[i]),
                           float((bounds[par][1] - ctr[i])/err[i])) if par in bounds else (None,None)
                          for i,par in enumerate(psr.pars())]

    res = scipy.optimize.minimize(func,[0.0]*len(ctr),method=method,**opts)

    if hasattr(res,'message'):
        print(res.message)

    # this will also set parameters to the minloc
    minchisq = func(res.x)

    if history:
        return minchisq, numpy.array(hval), numpy.array(hloc)
    else:       
        return minchisq

def glsfit(psr,renormalize=True):
    """Solve local GLS problem using scipy.linalg.cholesky.
    Update psr[...].val and psr[...].err from solution.
    If renormalize=True, normalize each design-matrix column by its norm."""
    
    res, err = psr.residuals(removemean=False)[psr.deleted == 0], psr.toaerrs[psr.deleted == 0]
    M = psr.designmatrix(updatebats=False,fixunits=True,fixsigns=True)
        
    C = numpy.diag((err * 1e-6)**2)
    
    if renormalize:
        norm = numpy.sqrt(numpy.sum(M**2,axis=0))
        M /= norm
    else:
        norm = 1.0
        
    mtcm = numpy.dot(M.T,numpy.dot(numpy.linalg.inv(C),M))
    mtcy = numpy.dot(M.T,numpy.dot(numpy.linalg.inv(C),res))
    
    xvar = numpy.linalg.inv(mtcm)
        
    c = scipy.linalg.cho_factor(mtcm)
    xhat = scipy.linalg.cho_solve(c,mtcy)

    sol = psr.vals()
    psr.vals(sol + xhat[1:] / norm[1:])
    psr.errs(numpy.sqrt(numpy.diag(xvar)[1:]) / norm[1:])
    
    return chisq(psr)
