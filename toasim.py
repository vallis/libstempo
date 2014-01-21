import math, os
import numpy as N

day = 24 * 3600
year = 365.25 * day

def make_ideal(psr):
    """Adjust the TOAs so that the residuals to zero, then refit."""
    
    psr.stoas[:] -= psr.residuals() / 86400.0
    psr.fit()

def add_efac(psr,efac=1.0):
    """Add nominal TOA errors, multiplied by `efac` factor."""
    
    psr.stoas[:] += efac * psr.toaerrs * (1e-6 / day) * N.random.randn(psr.nobs)

def add_equad(psr,equad):
    """Add quadrature noise of rms `equad` [s]."""
    
    psr.stoas[:] += (equad / day) * N.random.randn(psr.nobs)

def quantize(times,dt=1):
    bins    = N.arange(N.min(times),N.max(times)+dt,dt)
    indices = N.digitize(times,bins) # indices are labeled by "right edge"
    counts  = N.bincount(indices,minlength=len(bins)+1)

    bign, smalln = len(times), N.sum(counts > 0)

    t = N.zeros(smalln,'d')
    U = N.zeros((bign,smalln),'d')

    j = 0
    for i,c in enumerate(counts):
        if c > 0:
            U[indices == i,j] = 1
            t[j] = N.mean(times[indices == i])
            j = j + 1
    
    return t, U

def quantize_fast(times,dt=1):
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

# check that the two versions match
# t, U = quantize(N.array(psr.toas(),'d'),dt=1)
# t2, U2 = quantize_fast(N.array(psr.toas(),'d'),dt=1)
# print N.sum((t - t2)**2), N.all(U == U2)

def add_jitter(psr,equad,coarsegrain=0.1):
    """Add correlated quadrature noise of rms `equad` [s],
    with coarse-graining time `coarsegrain` [days]."""
    
    t, U = quantize_fast(N.array(psr.toas(),'d'),0.1)
    psr.stoas[:] += (equad / day) * N.dot(U,N.random.randn(U.shape[1]))

def add_rednoise(psr,A,gamma,components=10):
    """Add red noise with P(f) = A^2 / (12 pi^2) (f year)^-gamma,
    using `components` Fourier bases."""
    
    t = psr.toas()
    minx, maxx = N.min(t), N.max(t)
    x = (t - minx) / (maxx - minx)
    T = (day/year) * (maxx - minx)

    size = 2*components
    F = N.zeros((psr.nobs,size),'d')
    f = N.zeros(size,'d')

    for i in range(components):
        F[:,2*i]   = N.cos(2*math.pi*(i+1)*x)
        F[:,2*i+1] = N.sin(2*math.pi*(i+1)*x)

        f[2*i] = f[2*i+1] = (i+1) / T

    norm = A**2 * year**2 / (12 * math.pi**2 * T)
    prior = norm * f**(-gamma)
    
    y = N.sqrt(prior) * N.random.randn(size)
    psr.stoas[:] += (1.0/day) * N.dot(F,y)
    
def add_line(psr,f,A,offset=0.5):
    """Add a line of frequency `f` [Hz] and amplitude `A` [s],
    with origin at a fraction `offset` through the dataset."""
    
    t = psr.toas()
    t0 = offset * (N.max(t) - N.min(t))
    sine = A * N.cos(2 * math.pi * f * day * (t - t0))

    psr.stoas[:] += sine / day
