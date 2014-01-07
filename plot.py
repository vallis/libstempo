import types
import numpy as N
import matplotlib.pyplot as P

# select parameters by name or number, omit non-existing
def _select(p,pars,select):
    sel = []
    for s in select:
        if isinstance(s,str) and s in pars:
            sel.append(pars.index(s))
        elif isinstance(s,int) and s < p:
            sel.append(s)
    return len(sel), sel

def plothist(data,pars=[],offsets=[],norms=[],select=[],weights={},ranges={},append=False,
             bins=50,color='k',linestyle=None,title=None):
    if hasattr(data,'data') and not isinstance(data,N.ndarray):
        # parse a multinestdata structure
        if not pars and hasattr(data,'parnames'):
            pars = data.parnames
        data = data.data

    p = data.shape[-1]

    if not pars:
        pars = map('p{0}'.format,range(p))

    if offsets:
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
        # need to do this since isinstance(False,int) == True
        q = append if type(append) == int else p
        P.subplot(int((q-1)/4)+1,min(q,4),i+1)

        if append:
            P.hold(True)

        if pars[i] in ranges:
            dx = ranges[pars[i]]
            P.hist(data[:,i],bins=int(bins * (N.max(data[:,i]) - N.min(data[:,i])) / (dx[1] - dx[0])),
                   weights=weight,normed=True,histtype='step',color=color,linestyle=linestyle)
            P.xlim(dx)
        else:
            P.hist(data[:,i],bins=bins,
                   weights=weight,normed=True,histtype='step',color=color,linestyle=linestyle)

        P.xlabel(pars[i])
        P.ticklabel_format(style='sci',axis='x',scilimits=(-2,2),useoffset='True')
        P.hold(False)

    if title and not append:
        P.suptitle(title)

    P.tight_layout()

# to do: should fix this histogram so that the contours are correct
#        even for restricted ranges...
def _plotonehist2(x,y,parx,pary,smooth,ranges={},bins=50,weights=None):
    hrange = [ranges[parx] if parx in ranges else [N.min(x),N.max(x)],
              ranges[pary] if pary in ranges else [N.min(y),N.max(y)]]

    [h,xs,ys] = N.histogram2d(x,y,bins=bins,normed=True,range=hrange,weights=weights)
    P.contourf(0.5*(xs[1:]+xs[:-1]),0.5*(ys[1:]+ys[:-1]),h.T,cmap=P.get_cmap('YlOrBr')); P.hold(True)

    H,tmp1,tmp2 = N.histogram2d(x,y,bins=bins,range=hrange,weights=weights)

    if smooth:
        # only need scipy if we're smoothing
        import scipy.ndimage.filters as SNF
        H = SNF.gaussian_filter(H,sigma=1.5)

    if weights is None:
        H = H / len(x)
    else:
        H = H / N.sum(H)            # I think this is right...
    Hflat = -N.sort(-H.flatten())   # sort highest to lowest
    cumprob = N.cumsum(Hflat)       # sum cumulative probability

    levels = [N.interp(level,cumprob,Hflat) for level in (0.6826,0.9547,0.9973)]

    xs = N.linspace(hrange[0][0],hrange[0][1],bins)
    ys = N.linspace(hrange[1][0],hrange[1][1],bins)

    P.contour(xs,ys,H.T,levels,colors='k',linestyles=('-','--','-.'),linewidths=2); P.hold(False)

    if parx in ranges:
        P.xlim(ranges[parx])
    if pary in ranges:
        P.ylim(ranges[pary])

    P.xlabel(parx); P.ylabel(pary)
    P.ticklabel_format(style='sci',axis='both',scilimits=(-2,2),useoffset='True')

def plothist2(data,pars=[],offsets=[],smooth=False,select=[],ranges={},bins=50,diagonal=True,title=None,append=False):
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
                       normed=True,histtype='step',color='k')
                P.xlim(dx)
            else:
                P.hist(data[i],bins=50,normed=True,histtype='step',color='k')

            P.xlabel(pars[i])
            P.ticklabel_format(style='sci',axis='both',scilimits=(-2,2),useoffset='True')

            for j in range(0,i):
                if not append:
                    P.subplot(m,m,i*m+j+1)

                _plotonehist2(data[j],data[i],pars[j],pars[i],smooth,ranges,bins)
    else:
        for i in range(m-1):
            for j in range(i+1,m):
                if not append:
                    P.subplot(m-1,m-1,(m-1)*i+j)

                _plotonehist2(data[j],data[i],pars[j],pars[i],smooth,ranges,bins)

    P.tight_layout()

    if title and not append:
        P.suptitle(title)
    elif title:
        P.title(title)

    # if save:
    #     P.savefig('figs/{0}-{1}-2.png'.format(psr,flms[0]))
