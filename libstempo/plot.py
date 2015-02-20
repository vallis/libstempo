from __future__ import print_function

import math, types
import numpy as N
import matplotlib.pyplot as P

def plotres(psr,deleted=False,group=None):
    """Plot residuals, compute unweighted rms residual."""

    res, t, errs = psr.residuals(), psr.toas(), psr.toaerrs
    
    if (not deleted) and N.any(psr.deleted != 0):
        res, t, errs = res[psr.deleted == 0], t[psr.deleted == 0], errs[psr.deleted == 0]
        print("Plotting {0}/{1} nondeleted points.".format(len(res),psr.nobs))

    meanres = math.sqrt(N.mean(res**2)) / 1e-6
    
    if group is None:
        i = N.argsort(t)
        P.errorbar(t[i],res[i]/1e-6,yerr=errs[i],fmt='x')
    else:
        if (not deleted) and N.any(psr.deleted):
            flagmask = psr.flagvals(group)[~psr.deleted]
        else:
            flagmask = psr.flagvals(group)

        unique = list(set(flagmask))
            
        for flagval in unique:
            f = (flagmask == flagval)
            flagres, flagt, flagerrs = res[f], t[f], errs[f]
            i = N.argsort(flagt)
            P.errorbar(flagt[i],flagres[i]/1e-6,yerr=flagerrs[i],fmt='x')
        
        P.legend(unique,numpoints=1,bbox_to_anchor=(1.1,1.1))

    P.xlabel('MJD'); P.ylabel('res [us]')
    P.title("{0} - rms res = {1:.2f} us".format(psr.name,meanres))

# select parameters by name or number, omit non-existing
def _select(p,pars,select):
    sel = []
    for s in select:
        if isinstance(s,str) and s in pars:
            sel.append(pars.index(s))
        elif isinstance(s,int) and s < p:
            sel.append(s)
    return len(sel), sel

def plothist(data,pars=[],offsets=[],norms=[],select=[],weights={},ranges={},labels={},skip=[],append=False,
             bins=50,color='k',linestyle=None,linewidth=1,title=None):
    if hasattr(data,'data') and not isinstance(data,N.ndarray):
        # parse a multinestdata structure
        if not pars and hasattr(data,'parnames'):
            pars = data.parnames
        data = data.data

    p = data.shape[-1]

    if not pars:
        pars = map('p{0}'.format,range(p))

    if offsets:
        data = data.copy()

        if isinstance(offsets,dict):
            for i,par in enumerate(pars):
                if par in offsets:
                    data[:,i] = data[:,i] - offsets[par]
        else:
            if len(offsets) < p:
                offsets = offsets + [0.0] * (p - len(offsets))
            data = data - N.array(offsets)

    if norms:
        if len(norms) < p:
            norms = norms + [1.0] * (p - len(norms))

        data = data / norms

    if select:
        p, sel = _select(p,pars,select)
        data, pars = data[:,sel], [pars[s] for s in sel]

    if weights:
        weight = 1
        for i,par in enumerate(pars):
            if par in weights:
                if isinstance(weights[par],types.FunctionType):
                    weight = weight * N.vectorize(weights[par])(data[:,i])
                else:
                    weight = weight * weights[par]
    else:
        weight = None

    # only need lines for multiple plots
    # lines = ['dotted','dashdot','dashed','solid']

    if not append:
        P.figure(figsize=(16*(min(p,4)/4.0),3*(int((p-1)/4)+1)))

    for i in range(p):
        # figure out how big the multiplot needs to be
        if type(append) == int:                # need this since isinstance(False,int) == True
            q = append
        elif isinstance(append,(list,tuple)):
            q = len(append)
        else:
            q = p
        
        # increment subplot index if we're skipping
        sp = i + 1
        for s in skip:
            if i >= s:
                sp = sp + 1

        # if we're given the actual parnames of an existing plot, figure out where we fall
        if isinstance(append,(list,tuple)):
            try:
                sp = append.index(pars[i]) + 1
            except ValueError:
                continue

        P.subplot(int((q-1)/4)+1,min(q,4),sp)

        if append:
            P.hold(True)

        if pars[i] in ranges:
            dx = ranges[pars[i]]
            P.hist(data[:,i],bins=int(bins * (N.max(data[:,i]) - N.min(data[:,i])) / (dx[1] - dx[0])),
                   weights=weight,normed=True,histtype='step',color=color,linestyle=linestyle,linewidth=linewidth)
            P.xlim(dx)
        else:
            P.hist(data[:,i],bins=bins,
                   weights=weight,normed=True,histtype='step',color=color,linestyle=linestyle,linewidth=linewidth)

        P.xlabel(labels[pars[i]] if pars[i] in labels else pars[i])

        # P.ticklabel_format(style='sci',axis='both',scilimits=(-3,4),useoffset='True')
        P.locator_params(axis='both',nbins=6)
        P.minorticks_on()

        fx = P.ScalarFormatter(useOffset=True,useMathText=True)
        fx.set_powerlimits((-3,4)); fx.set_scientific(True)

        fy = P.ScalarFormatter(useOffset=True,useMathText=True)
        fy.set_powerlimits((-3,4)); fy.set_scientific(True)

        P.gca().xaxis.set_major_formatter(fx)
        P.gca().yaxis.set_major_formatter(fy)

        P.hold(False)

    if title and not append:
        P.suptitle(title)

    P.tight_layout()

# to do: should fix this histogram so that the contours are correct
#        even for restricted ranges...
def _plotonehist2(x,y,parx,pary,smooth=False,colormap=True,ranges={},labels={},bins=50,levels=3,weights=None,
                  color='k',linewidth=1):
    hold = P.ishold()

    hrange = [ranges[parx] if parx in ranges else [N.min(x),N.max(x)],
              ranges[pary] if pary in ranges else [N.min(y),N.max(y)]]

    [h,xs,ys] = N.histogram2d(x,y,bins=bins,normed=True,range=hrange,weights=weights)
    if colormap:
        P.contourf(0.5*(xs[1:]+xs[:-1]),0.5*(ys[1:]+ys[:-1]),h.T,cmap=P.get_cmap('YlOrBr')); P.hold(True)

    H,tmp1,tmp2 = N.histogram2d(x,y,bins=bins,range=hrange,weights=weights)

    if smooth:
        # only need scipy if we're smoothing
        import scipy.ndimage.filters as SNF
        H = SNF.gaussian_filter(H,sigma=1.5 if smooth is True else smooth)

    if weights is None:
        H = H / len(x)
    else:
        H = H / N.sum(H)            # I think this is right...
    Hflat = -N.sort(-H.flatten())   # sort highest to lowest
    cumprob = N.cumsum(Hflat)       # sum cumulative probability

    levels = [N.interp(level,cumprob,Hflat) for level in [0.6826,0.9547,0.9973][:levels]]

    xs = N.linspace(hrange[0][0],hrange[0][1],bins)
    ys = N.linspace(hrange[1][0],hrange[1][1],bins)

    P.contour(xs,ys,H.T,levels,
              colors=color,linestyles=['-','--','-.'][:len(levels)],linewidths=linewidth)
    P.hold(hold)

    if parx in ranges:
        P.xlim(ranges[parx])
    if pary in ranges:
        P.ylim(ranges[pary])

    P.xlabel(labels[parx] if parx in labels else parx)
    P.ylabel(labels[pary] if pary in labels else pary)

    P.locator_params(axis='both',nbins=6)
    P.minorticks_on()

    fx = P.ScalarFormatter(useOffset=True,useMathText=True)
    fx.set_powerlimits((-3,4)); fx.set_scientific(True)

    fy = P.ScalarFormatter(useOffset=True,useMathText=True)
    fy.set_powerlimits((-3,4)); fy.set_scientific(True)

    P.gca().xaxis.set_major_formatter(fx)
    P.gca().yaxis.set_major_formatter(fy)

def plothist2(data,pars=[],offsets=[],smooth=False,colormap=True,select=[],ranges={},labels={},bins=50,levels=3,weights=None,cuts=None,
              diagonal=True,title=None,color='k',linewidth=1,append=False):
    if hasattr(data,'data') and not isinstance(data,N.ndarray):
        # parse a multinestdata structure
        if not pars    and hasattr(data,'parnames'):
            pars    = data.parnames
        data = data.data

    m = data.shape[-1]

    if not pars:
        pars = map('p{0}'.format,range(m))

    if offsets:
        if len(offsets) < m:
            offsets = offsets + [0.0] * (m - len(offsets))
        data = data - N.array(offsets)

    if cuts:
        for i,par in enumerate(pars):
            if par in cuts:
                data = data[data[:,i] > cuts[par][0],:]
                data = data[data[:,i] < cuts[par][1],:]

    if weights:
        weight = 1
        for i,par in enumerate(pars):
            if par in weights:
                if isinstance(weights[par],types.FunctionType):
                    weight = weight * N.vectorize(weights[par])(data[:,i])
                else:
                    weight = weight * weights[par]
    else:
        weight = None

    if select:
        m, sel = _select(m,pars,select)
        data, pars = data[:,sel], [pars[s] for s in sel]

    if not append:
        fs = min((m if diagonal else m-1)*4,16)
        P.figure(figsize=(fs,fs))

    data = data.T

    if diagonal:
        for i in range(m):
            if not append:
                P.subplot(m,m,i*(m+1)+1)

            if pars[i] in ranges:
                dx = ranges[pars[i]]
                P.hist(data[i],bins=int(50 * (N.max(data[i]) - N.min(data[i])) / (dx[1] - dx[0])),
                       weights=weight,normed=True,histtype='step',color='k')
                P.xlim(dx)
            else:
                P.hist(data[i],bins=50,weights=weight,normed=True,histtype='step',color='k')

            P.xlabel(labels[pars[i]] if pars[i] in labels else pars[i])
            P.ticklabel_format(style='sci',axis='both',scilimits=(-2,2),useoffset='True')
            # P.tick_params(labelsize=12)

            for j in range(0,i):
                if not append:
                    P.subplot(m,m,i*m+j+1)

                _plotonehist2(data[j],data[i],pars[j],pars[i],smooth,colormap,ranges,labels,bins,levels,weights=weight,
                              color=color,linewidth=linewidth)
    else:
        for i in range(m-1):
            for j in range(i+1,m):
                if not append:
                    P.subplot(m-1,m-1,(m-1)*i+j)

                _plotonehist2(data[j],data[i],pars[j],pars[i],smooth,colormap,ranges,labels,bins,levels,weights=weight,
                              color=color,linewidth=linewidth)

    P.tight_layout()

    if title and not append:
        P.suptitle(title)
    elif title:
        P.title(title)

    # if save:
    #     P.savefig('figs/{0}-{1}-2.png'.format(psr,flms[0]))


def plotgwsrc(gwb):
    """
    Plot a GWB source population as a mollweide projection.
    """
    theta, phi, omega, polarization = gwb.gw_dist()

    rho = phi-N.pi
    eta = 0.5*N.pi - theta

    # I don't know how to get rid of the RuntimeWarning -- RvH, Oct 10, 2014:
    #     /Users/vhaaster/env/dev/lib/python2.7/site-packages/matplotlib/projections/geo.py:485:
    #     RuntimeWarning: invalid value encountered in arcsin theta = np.arcsin(y / np.sqrt(2))
    #old_settings = N.seterr(invalid='ignore')

    P.title("GWB source population")
    ax = P.axes(projection='mollweide')

    foo = P.scatter(rho, eta, marker='.', s=1)
    #bar = N.seterr(**old_settings)

    return foo
