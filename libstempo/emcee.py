import math
import numpy as N
from multinest import load_emcee as load

# note: we run emcee as if it was multinest, by remapping parameter ranges to [0,1]
#       if we didn't, the logPL function could just be
#
#       prior = p0.prior(xs)    
#       return -N.inf if not prior else math.log(prior) + ll.loglike(xs)
#
#       and the initialization would be
#
#       init = [p0.remap_list(N.random.random(len(cfg.searchpars))) for i in range(cfg.walkers)]

def logPL(ll,p0,xs):
    pprior = p0.premap(xs)

    # libstempo.like.Prior mappers are supposed to throw a ValueError
    # if they get coordinates out of range
    try:
        pars = p0.remap(xs)
    except ValueError:
        return -N.inf

    prior = pprior * p0.prior(pars)

    return -N.inf if not prior else math.log(prior) + ll.loglike(pars)

def save(basename,sample,p0,skip):
    # save last cloud to resume from it
    N.save(basename + '-resume.npy',sample.chain[:,-1,:])

    # thin out the run
    chain, lnprob = sample.chain[:,::skip,:], sample.lnprobability[:,::skip]

    N.save(basename + '-lnprob.npy',lnprob)

    N.save(basename + '-chain-unmapped.npy',chain)
    # remap parameters to physical ranges
    for w in range(chain.shape[0]):
        for s in range(chain.shape[1]):
            p0.premap(chain[w,s,:])
            chain[w,s,:] = p0.remap_list(chain[w,s,:])
    N.save(basename + '-chain.npy', chain)

def merge(data,skip=50,fraction=1.0):
    """Merge one every 'skip' clouds into a single emcee population,
    using the later 'fraction' of the run."""

    w,s,d = data.chains.shape
    
    start = int((1.0 - fraction) * s)
    total = int((s - start) / skip)
        
    return data.chains[:,start::skip,:].reshape((w*total,d))

def cull(data,index,min=None,max=None):    
    """Sieve an emcee clouds by excluding walkers with search variable 'index'
    smaller than 'min' or larger than 'max'."""

    ret = data

    if min is not None:
        ret = ret[ret[:,index] > min,:]

    if max is not None:
        ret = ret[ret[:,index] < max,:]
    
    return ret

