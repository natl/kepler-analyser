"""
model_analysis.py
Author: Nathanael Lampe, Monash University
Date Created: 19/7/2013 (Anniversary of the moon landing, woohoo)

Purpose:
To recode old IDL routines that took KEPLER models and produced mean
lightcurves and burst parameters.

Changelog:
Modified in August 2016 by NL to
- Add compatibility with lmfit >v0.9
- Migrate to four space tabs and improve PEP8 compliance for code linters

Modified in July 2016 by LK to
- remove obsolete routines
- load from binary files
- add a 'qb' array to the keyTable
- refactored code to be object oriented
"""

from __future__ import division, unicode_literals, print_function
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
from lmfit import minimize, Parameters
import bottleneck as bn
import os
import logging

np.seterr(divide='raise', over='warn', under='warn', invalid='raise')

###############################################################################
###############################################################################
#                                                                             #
#                      ANALYSIS SUBROUTINES                                   #
#                                                                             #
###############################################################################
###############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FITTING BURST TAILS~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def fitTail(time, lumin, dlumin, params, method='power'):
    '''
    Fit a function to the burst tail

    params, fitData = fitTail(time, lumin, method='power')
    =============================================================================

    args:
    -----
    time: array of times in burst tail
    lumin: array of luminositiws through burst tail
    dlumin: array of luminosity uncertainties
    params: lmfit.Parameters object specifying parameters [Note 1]

    kwargs:
    -----
    method: type of function to fit, options are
        'power': fit a power law like in [1]
        'exp1':    fit a single exponential [1]
        'exp2':    fit a double exponential (not implemented)

    returns:
    -----
    params:    lmfit.Parameters object with fit parameters
    fitData: dictionary with fit statistics

    =============================================================================
    Notes:

    1. The lmfit.Parameters class object specifies parameter values and bounds.
             The parameters it needs to specify vary depending on the fitting
             method,
             -----
             exp1 lmfit.parameters Object contains [F0, t0, tau, Fp]
                 F0 = initial flux
                 t0 = time shift
                 tau = time constant
                 Fp = persistent emission
             -----
             power lmfit.Parameters Object contains [F0, t0, ts, al, Fp]
                F0 = initial flux
                t0 = time shift (decay starts)
                ts = time shift (burst starts)
                al = power law exponent
                Fp = persistent emission
             -----

    =============================================================================
    Changelog:
    Created 22-11-2013
    26-11-2013: Added single tail and power law functions, work on chi-square
                            minimisation
    18-02-2014: Added fit statistics to be returned as a dictionary
    03-03-2014: Fit from 90% to 10% in the tail, rather than be adaptive.

    =============================================================================
    References:
    [1] in't Zand et al., The cooling rate of neutron stars after thermonuclear
            shell flashes, A&A (submitted 2013)

    '''

    # opening assertions
    assert len(time) == len(lumin), 'time and lumin should be same length'
    assert np.ndim(time) == 1, 'time & lumin should be 1-d'
    assert method in ['power', 'exp1', 'exp2'], 'method kwarg invalid'

    # copy arrays
    time = deepcopy(np.asarray(time))
    lumin = deepcopy(np.asarray(lumin))
    dlumin = deepcopy(np.asarray(dlumin))

    def exp1(t, p):
        '''
        Single Tail Exponential
        exp1(t, params)
        ===========================================================================
        args:
        -----
            t: 1-d array of burst times
            p: lmfit.parameters Object [F0, t0, tau, Fp]
                F0 = initial flux
                t0 = time shift
                tau = time constant
                Fp = persistent emission

        returns:
        -----
        1-D array of burst luminosities as predicted by a single tail
            exponential with parameters set by params in p

        '''
        F0 = p['F0'].value
        t0 = p['t0'].value
        tau = p['tau'].value
        Fp = p['Fp'].value

        return F0*np.exp(-(t-t0)/tau) + Fp

    def power(t, p):
        '''
        Power Law
        power(t, params)
        ===========================================================================
        args:
        -----
            t: 1-d array of burst times
            p: lmfit.Parameters Object with [F0, t0, ts, al, Fp]
                F0 = initial flux
                t0 = time shift (decay starts)
                ts = time shift (burst starts)
                al = power law exponent
                Fp = persistent emission

        returns:
        -----
        1-D array of burst luminosities as predicted by a power law
            with parameters set by params in p
        '''

        F0 = p['F0'].value
        t0 = p['t0'].value
        ts = p['ts'].value
        al = p['al'].value
        Fp = p['Fp'].value
        return F0*((t-ts)/(t0-ts))**(-1.0*al) + Fp

    msg = 'Two tail exponential not available yet'

    if method == 'power':
        f = power
    elif method == 'exp1':
        f = exp1
    elif method == 'exp2':
        raise NotImplementedError(msg)
    else:
        raise SystemError('Method kwarg incorrect')

    def minFunc(p, t, l, dl, f):
        '''
        Minimisation Function
        '''
        lm = f(t, p)
        csq = ((l - lm)/dl)
        return csq

    result = minimize(minFunc, params, args=(time, lumin, dlumin, f),
                      method='leastsq')
    fitData = {'nfev': result.nfev,  # number of function evaluations
               'success': result.success,  # boolean for fit success
               'errorbars': result.errorbars,  # boolean, errors were estimated
               'message': result.message,  # message about fit success.
               'ier': result.ier,  # int error value from sp.optimize.leastsq
               'lmdif_message': result.lmdif_message,  # from sp.opt.leastsq
               'nvarys': result.nvarys,  # number of variables in fit
               'ndata': result.ndata,  # number of data points
               'nfree': result.nfree,  # degrees of freedom in fit
               'residual': result.residual,  # residual array /return of func()
               'chisqr': result.chisqr,  # chi-square
               'redchi': result.redchi}  # reduced chi-square

    return result.params, fitData

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fitting routine ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def burstFits(mt, ml, pt, mdl=None):
    '''
    Conduct a least squares minimisation of a power law and single tail
    exponential on a thermonuclear burst tail

    pParams,pData,eParams,eData = burstFits(mt, ml, pt, mdl = None)
    =============================================================================

    args:
    -----
    mt: Tail time (since burst start, not from tail start)
    ml: Tail luminosity
    pt: Time of burst Peak

    kwargs:
    -----
    mdl: Optionally weight the points by uncertainty (chi-square minimisation)
             if None, assume uniform weights

    returns:
    -----
    pParams: lmfit.Parameters object with power law fit final parameters
    pData:     dictionary with fit statistics for pLaw fit
    eParams: lmfit.Parameters object with exponential fit final parameters
    eData:     dictionary with fit statistics for exp law fit
    In both cases, pp['rcs']

    =============================================================================
    Notes:
    Conducts fits to burst tails following In't Zand et al, A&A 562 A16 (2014),
    except fit from 0.90Fpeak to 0.10Fpeak, rather than adaptively choose where
    to start fitting.
    '''

    peakTime = bn.nanargmax(mt > pt)
    peakLum = ml[peakTime]
    persLum = ml[0]/peakLum  # (scaled persLum)

    # rescaling, by 1/peakLum, and set equal weights if mdl == None
    if mdl is None:
        t = mt[peakTime:]
        l = ml[peakTime:]/peakLum
        dl = np.ones_like(t)
    else:
        # remove artificially low uncertainties
        ok = np.nonzero((mdl > 1e35) & (mt > 0))[0]
        t = mt[ok]
        l = ml[ok]/peakLum
        dl = mdl[ok]/peakLum

    # find t90, t10

    n = len(l)
    # last occurence crossing 0.1Lpeak (note >)
    t10Ind = n - bn.nanargmax(l[::-1] > 0.1)
    # find last point where we have 0.9Lpeak (note >=)
    t90Ind = n - bn.nanargmax(l[::-1] >= 0.9)
    t0 = t[t90Ind]
    tFit = t[t90Ind:t10Ind]
    lFit = l[t90Ind:t10Ind]
    dlFit = dl[t90Ind:t10Ind]

    p = Parameters()
    p.add('F0', value=1., vary=True)
    p.add('t0', value=t0, vary=False)
    p.add('ts', value=-1, vary=True, min=-10, max=t0)
    p.add('al', value=1.4, vary=True, min=0)
    p.add('Fp', value=persLum, vary=True, min=0., max=1.)
    pParams, pData = fitTail(tFit, lFit, dlFit, p, method='power')

    p = Parameters()
    p.add('F0', value=1., vary=True)
    p.add('t0', value=t0, vary=False)
    p.add('tau', value=10, vary=True, min=0)
    p.add('Fp', value=persLum, vary=True, min=0., max=1.)
    eParams, eData = fitTail(tFit, lFit, dlFit, p, method='exp1')

    # rescale back
    for i in (pParams['F0'], pParams['Fp'], eParams['F0'], eParams['Fp']):
            if i.value != 0:
                    i.value = i.value*peakLum
            if i.stderr != 0:
                    i.stderr = i.value*peakLum
    return pParams, pData, eParams, eData


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~RISE TIMES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def findAtRiseFrac(time, lumin, peakLum, persLum, frac):
    '''
    findAtRiseFrac(time, lumin, peakLum, persLum, frac)

    args:
        time: 1-d array of burst rise times
        lumin: 1-d array of burst rise luminosities
        peakLum: peak luminosity
        persLum: persistent luminosity
        frac: fraction of peak luminsoity to find time at

    return:
        float, time at which burst reaches frac*(peakLum-persLum)

    A function to find the time a burst reaches a certain luminosity as
    a fraction of the peak luminosity, screening out spurious results from
    convective shocks.

    Created November 15, 2013
    '''

    time = np.asarray(deepcopy(time))
    lumin = np.asarray(deepcopy(lumin))

    lumin -= persLum
    peakLum -= persLum
    fracLum = frac*peakLum

    # find candidates where we cross the required Lumfrac
    greater = lumin > fracLum
    cross = greater - np.roll(greater, 1)
    cands = np.nonzero(cross[1:])[0] + 1  # candidates to cross fracLum

    avgBurstGrad = (lumin[-1]-lumin[0])/(time[-1]-time[0])

    grads = (lumin[cands]-lumin[cands-1])/(time[cands]-time[cands-1])
    # exclude -ve gradients (added apr-03-2014)
    for ii in xrange(len(grads)):
        if grads[ii] < 0:
            grads[ii] = np.Inf

    # calculate gradient by back-step FD, rather than central diff
    candGrads = zip(cands, grads)
    candGrads.sort(key=lambda x: x[1])  # sort by gradient

    best = candGrads[0][0]  # Most likely candidate has lowest gradient

    # Now interpolate for more precision
    interpTimes = np.linspace(time[best-1], time[best+1], 2**6)
    lumInterp = interp1d(time, lumin, kind='linear', copy=True)
    interpLums = lumInterp(interpTimes)
    interpBest = bn.nanargmax(interpLums > fracLum)

    return interpTimes[interpBest]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~SMOOTHING LIGHTCURVES~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def smooth(t, f, r, tmin):
    '''
    newt, newf, newr = smooth(t,f,r,tmin)

    args:
        t = 1-D array of times
        f = 1-D array of fluxes
        r = 1-D array of radii
        tmin = float, new minimum time bin size

    returns:
        newt = smoothed t values
        newf = smoothed f values
        newr = smoothed r values

    A routine to take a lightcurve and return a new lightcurve that averages
    over variations that last less than tmin.
    Specifically designed to be used on lightcurves that show large shocks e.g.
    model a6. In these cases, a number of millisecond shocks cause large
    increases in stellar luminosity due to a convective layer reaching the
    stellar surface. These are too small to typically be observed, and boggle
    the analysis routine that tries to identify peak luminosities
    '''
    # some simple asseritons
    assert len(t) == len(r) and len(t) == len(f),\
        '''t,r,f should be the same shape'''
    assert len(np.shape(t)) == 1, '''t should be 1-D'''
    # set up lists, loops
    ii = 0
    newt = []
    newf = []
    newr = []
    dt = t - np.roll(t, 1)
    n = len(t)

    while ii < n:
        if dt[ii] > tmin:
            # case 1: bigger timestep than tmin
            newt.append(t[ii])
            newf.append(f[ii])
            newr.append(r[ii])
            ii += 1
        elif dt[ii] < 0:
            # case 2: first timestep will have dt<0
            newt.append(t[ii])
            newf.append(f[ii])
            newr.append(r[ii])
            ii += 1
        else:
            # case 3: smaller timesteps than tmin
            tStart = ii
            tStop = max(tStart, np.argmax(t > (t[ii] + tmin)))
            if (tStop != tStart) and tStop != n:
                # 3a: not at the EOF
                newt.append(t[ii]+tmin/2.)
                avgf = sum(f[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])  # NOQA
                avgr = sum(r[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])  # NOQA
                newf.append(avgf)
                newr.append(avgr)
                ii = tStop
            elif (tStop != tStart):
                # 3b found the EOF, average to EOF
                tStop = n-1
                newt.append((t[tStart] + t[tStop]) / 2.)
                avgf = sum(f[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])  # NOQA
                avgr = sum(r[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])  # NOQA
                newf.append(avgf)
                newr.append(avgr)
                ii = n
            else:
                ii = n  # Forget about the last lonely point
    return np.array(newt), np.array(newf), np.array(newr)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Shock Checking~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def isShock(times, fluxes, index):
    '''
    isShock(times, fluxes, index)

    args:
        times = 1-d array of times
        fluxes = 1-d array of fluxes
        index = int, index to check
    returns:
        True if index is a shock
        False if index is not a shock

    used to identify shocks in the burst rise, so they are not treated as a
    maximum by the analysis routine

    shocks appear in the rise due to a convective zone hitting the surface of
    the neutron star atmosphere, causing a swelling in luminosity. They are
    not usually observable as burning as not usually spherically symmetric, and
    they have a very short duration

    '''
    # some assertions
    assert len(np.shape(times)) == 1, ''' times should be 1-d'''
    assert index == np.int_(index), '''index should be type int'''
    assert np.shape(times) == np.shape(fluxes), '''times and fluxes should have
        the same shape'''

    # shocks do not change the luminosity much either side of the shock, but
    # are vastly different to the shock
    # get indices for +/- 0.05s from shock

    justPrev = bn.nanargmax(times > (times[index] - 0.02))
    if justPrev == index:
        justPrev -= 1
    justAfter = bn.nanargmax(times > (times[index] + 0.02))
    if justAfter == index:
        justAfter += 1

    if ((fluxes[justPrev]/fluxes[index] > 0.8) or
            (fluxes[justAfter]/fluxes[index] > 0.8)):
        # we have no shock, as the values either side of this point suggest
        # a flat top, or is a step
        if fluxes[index+1] < 0.95*fluxes[index]:
            # extra little routine to pick up shocks near the burst peak
            shock = True
        else:
            shock = False
    else:
        shock = True

    return shock

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Simpson's Rule~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def simprule(a, b, c, dt):
    '''
    simprule(a,b,c,dt)
    evaluate the area underneath three points by Simpson's Rule
    the three points ought to be equally spaced, with b being the midpoint
    the inputs a,b,and c are f(start), f(mid) and f(end) respectively,
    dt is the spacing

    Thus, Simpson's rule becomes here: area=([c-a]/6)(f(a)+4f(b)+f(c))
    '''
    return (a + 4.0*b + c)*dt/6.0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~Check Local Maximum~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def isMaximum(a, b, c):
    '''
    Small routine to check if point b is a local maximum
    '''
    if b > a and b > c:
        return True
    else:
        return False

###############################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Convexity~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
###############################################################################


def convexity(time, lumin, peakLum, persLum):
    '''
    convexity(time, lumin, peakLum, persLum)

    args:
        time: 1-d array of times (burst rise)
        lumin: 1-d array of luminosities
        peakLum: peak luminosity of the burst
        persLum: persistent luminosity of the burst

    returns:
        convexity parameter according to [1] as a float

    calculate the convexity of a thermonuclear burst from the burst rise
    This algorithm will flip out if you use a whole burst.

    References:
        [1] Maurer and Watts (2008), MNRAS, 383, p387
    '''
    time = np.asarray(deepcopy(time))
    lumin = np.asarray(deepcopy(lumin))

    lumin -= persLum
    peakLum -= persLum  # correct for persistent luminosity

    n = len(lumin)

    gt10 = lumin > 0.1*peakLum
    ten = (n) - bn.nanargmin(gt10[::-1])
    gt90 = lumin > 0.9*peakLum
    ninety = (n) - bn.nanargmin(gt90[::-1])

    # for 10%
    fa = 0.1*peakLum
    m = (lumin[ten] - lumin[ten-1])/(time[ten] - time[ten-1])
    ta = time[ten-1] + (fa - lumin[ten-1])/m

    # for 90%
    fb = 0.9*peakLum
    m = (lumin[ninety] - lumin[ninety-1])/(time[ninety] - time[ninety-1])
    tb = time[ninety-1] + (fb - lumin[ninety-1])/m

    t = np.concatenate(([ta], time[ten:ninety], [tb]))
    l = np.concatenate(([fa], lumin[ten:ninety], [fb]))

    lRange = np.max(l) - np.min(l)
    if lRange != 0:
        t -= t[0]
        l -= l[0]
        lastl = l[-1]
        lastt = t[-1]
        t *= (10./lastt)
        l *= (10./lastl)

        c = 0
        for ii in xrange(1, len(t)):
            c += 0.5*((l[ii] + l[ii-1]) - (t[ii] + t[ii-1]))*(t[ii] - t[ii-1])
        return c
    else:
        return 0

###############################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LC Averaging~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
###############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Average to key burst~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def avgParams(times, lums, rads):
    '''
    sumParams(times, lums, rads)
    each paramater is a list of arrays to be averaged

    Arrays are averaged to the burst with the latest finishing time
    '''
    n = len(times)
    if n == 1:
        dl = np.zeros(len(times[0]))
        dr = np.zeros(len(times[0]))
        t = times[0]
        l = lums[0]
        r = rads[0]
    elif n == 2:
        lens = map(max, times)
        if lens[1] > lens[0]:
            # Average using longer array
            times.reverse()
            lums.reverse()
            rads.reverse()
        t = times[0]
        # l = 0.5*lums[0]
        # r = 0.5*rads[0]
        interpLum = interp1d(times[1], lums[1], fill_value=np.nan,
                             bounds_error=False)
        interpRad = interp1d(times[1], rads[1], fill_value=np.nan,
                             bounds_error=False)
        l = np.nanmean([lums[0], interpLum(t)], axis=0)  # 0.5*interpLum(t)
        r = np.nanmean([rads[0], interpRad(t)], axis=0)  # 0.5*interpRad(t)

        dl = np.zeros(len(t))
        dr = np.zeros(len(t))

    elif n >= 3:
        # exclude first burst, interpolate along 2, then average
        times = times[1:]
        lums = lums[1:]
        rads = rads[1:]
        lengths = map(max, times[1:])
        longest = np.argmax(lengths) + 1

        t = times[longest]
        interpolatedLums = [lums[longest]]
        interpolatedRads = [rads[longest]]

        for ii in range(0, n-1):
            if ii != longest:
                interpLum = interp1d(times[ii], lums[ii], fill_value=np.nan,
                                     bounds_error=False)
                interpRad = interp1d(times[ii], rads[ii], fill_value=np.nan,
                                     bounds_error=False)
                interpolatedLums.append(interpLum(t))
                interpolatedRads.append(interpRad(t))

        interpolatedLums = np.asarray(interpolatedLums, dtype=np.float64)
        interpolatedRads = np.asarray(interpolatedRads, dtype=np.float64)
        # can we do some outlier analysis here to remove shocks in the rise?
        # a question for later analyses.

        l = np.nanmean(interpolatedLums, axis=0)
        r = np.nanmean(interpolatedRads, axis=0)

        dl = np.nanstd(interpolatedLums, axis=0, ddof=0)
        dr = np.nanstd(interpolatedRads, axis=0, ddof=0)
        # now remove any sneaky NaN's that have gotten through
        # shouldn't have to do this, but apparently I do.
        # NaN's may arise from every column in interpolatedLums being NaN, but
        # we shouldn't interpolate here

        bad_dl = np.isnan(dl)
        bad_dr = np.isnan(dr)
        ok = np.logical_not(np.logical_or(bad_dl, bad_dr))
        t = t[ok]
        l = l[ok]
        r = r[ok]
        dl = dl[ok]
        dr = dr[ok]

        # There will still be spots where dl=0 (std of one value) but now
        # there are no NaN's!!
    return t, l, r, dl, dr

# ~~~~~~~~~~~~~~~~~~~~~~~~~Average to fixed interval~~~~~~~~~~~~~~~~~~~~~~~~~ #


def intervalAvg(times, lums, rads, interval=0.125):
    '''
    Average burst LC's onto a grid with a fixed observation interval
    sumParams(times, lums, rads, interval = 0.125)
    =============================================================================

    args
    -----
        times: list containing times for each burst
        lums: list containing luiminosities for each burst
        rads: list containing radii for each burst

    kwargs:
    -----
        interval: time resolution

    returns:
    -----
        t, l, r, dl, dr
        t: time
        l: mean luminosity
        r: mean radius
        dl: 1-sigma error in luminosity
        dr: 1-sigma error in radius
    ============================================================================

    Notes:
    This routine considers that observations integrate across 0.125s intervals
    in x-ray detectors.
    Hence, rather than binning to the model resolution, we consider integrating
    the observed flux in fixed width bins. This defaults to 0.125s, as this is
    the finest bin-width commonly encountered
    ============================================================================
    Changelog:
    Created 25-11-2013 from avgParams


    '''

    def interp(times, lums, rads, interval):
        '''
        Call this routine in multi-burst averaging
        '''
        # make new time array
        minTime = min([t[0] for t in times])
        maxTime = max([t[-1] for t in times])

        posTime = np.arange(0, maxTime, interval)
        negTime = np.arange(-1*interval, minTime, -1*interval)
        newTime = np.concatenate((negTime[::-1], posTime))

        # now we integrate across each bin
        newL = []
        newR = []

        for (t, l, r) in zip(times, lums, rads):

            newl = []  # define lists for new burst
            newr = []
            for tBin in newTime:
                # tbin is the bin midpoint for integration
                mint = tBin - interval/2
                maxt = tBin + interval/2
                goodIndex = np.nonzero(np.logical_and(t > mint, t < maxt))[0]

                if len(goodIndex) > 1:
                    mni = goodIndex[0]
                    mxi = goodIndex[-1]
                    try:
                        # if the points all exist, integrate
                        # startpoints
                        l0 = l[mni-1] + (l[mni]-l[mni-1])/(t[mni]-t[mni-1])*(mint-t[mni-1])  # NOQA
                        r0 = r[mni-1] + (r[mni]-r[mni-1])/(t[mni]-t[mni-1])*(mint-t[mni-1])  # NOQA
                        # first trapezoid
                        ll = 0.5*(l[mni]+l0)*(t[mni]-mint)
                        rr = 0.5*(r[mni]+r0)*(t[mni]-mint)
                        # middle trapezoids
                        for ind in goodIndex[1:]:
                            ll += 0.5*(l[ind]+l[ind-1])*(t[ind]-t[ind-1])
                            rr += 0.5*(r[ind]+r[ind-1])*(t[ind]-t[ind-1])
                        # endpoints
                        lf = l[mxi] + (l[mxi+1]-l[mxi])/(t[mxi+1]-t[mxi])*(maxt-t[mxi])  # NOQA
                        rf = r[mxi] + (r[mxi+1]-r[mxi])/(t[mxi+1]-t[mxi])*(maxt-t[mxi])  # NOQA
                        # last trapezoid
                        ll += 0.5*(l[mxi+1]+lf)*(maxt-t[mxi])
                        rr += 0.5*(r[mxi+1]+rf)*(maxt-t[mxi])
                        # normalise
                        ll /= interval
                        rr /= interval
                    except IndexError:
                        ll = np.NaN
                        rr = np.NaN
                elif len(goodIndex) in [1]:
                    ll = l[goodIndex[0]]
                    rr = r[goodIndex[0]]
                else:
                    idx = bn.nanargmax(t > tBin)
                    if idx != 0:
                        ll = l[idx-1] + (l[idx]-l[idx-1])/(t[idx]-t[idx-1])*(tBin-t[idx-1])  # NOQA
                        rr = r[idx-1] + (r[idx]-r[idx-1])/(t[idx]-t[idx-1])*(tBin-t[idx-1])  # NOQA
                    else:
                        ll = np.NaN
                        rr = np.NaN
                newl.append(ll)
                newr.append(rr)
            newL.append(np.array(newl))
            newR.append(np.array(newr))

        # and now take averages. Hooray
        t = newTime
        l = np.nanmean(np.asarray(newL), axis=0)
        r = np.nanmean(np.asarray(newR), axis=0)
        dl = np.nanstd(np.asarray(newL), axis=0, ddof=0)
        dr = np.nanstd(np.asarray(newR), axis=0, ddof=0)

        # now remove any sneaky NaN's that have gotten through
        # NaN's may arise from every column in interpolatedLums being NaN, but
        # we shouldn't interpolate here

        bad_dl = np.isnan(dl)
        bad_dr = np.isnan(dr)
        ok = np.logical_not(np.logical_or(bad_dl, bad_dr))
        t = t[ok]
        l = l[ok]
        r = r[ok]
        dl = dl[ok]
        dr = dr[ok]

        # There will still be spots where dl=0 (std of one value) but now there
        # are no NaN's!!
        return t, l, r, dl, dr

    # Commence routine
    n = len(times)

    if n == 1:
        # cannot really perform an average
        dl = np.zeros(len(times[0]))
        dr = np.zeros(len(times[0]))
        t = times[0]
        l = lums[0]
        r = rads[0]

    elif n == 2:
        # Average second burst with first
        t, l, r, dl, dr = interp(times, lums, rads, interval)

    elif n >= 3:
        # average bursts from second onwards
        t, l, r, dl, dr = interp(times[1:], lums[1:], rads[1:], interval)

    return t, l, r, dl, dr

###############################################################################
###############################################################################
#                                                                             #
#                      INDIVIDUAL BURST SEPARATION                            #
#                                                                             #
###############################################################################
###############################################################################


def separate(burstTime, burstLum, burstRad, modelID, outputDirectory):
    '''
    separate(bursttime, burstflux, burstrad, modelID, outputDirectory)
    This program is designed to separate the bursts in the models into
    individual burst files for subsequent analysis. It will take the delta t
    value's for each burst
    Individual burst files do not have the persistent luminosity subtracted
    '''
    print('SEPARATING '+str(modelID))
    beginTimeBackJump = 20  # seconds before burst to start recording
    peakTimeBackLook = 0.3  # look for the peak in a 5 second range
    peakTimeForwardLook = 5.0  # look forward 5s, and back less
    tim = np.asarray(burstTime, dtype=np.float64)
    lum = np.asarray(burstLum, dtype=np.float64)
    rad = np.asarray(burstRad, dtype=np.float64)
    minFlux = 1.e36

    # ANALYSIS

    jj = bn.nanargmax(tim > beginTimeBackJump) + 1
    startTime = np.argmax(tim > peakTimeBackLook)
    tDel = [0]
    alpha = [0]
    peakTime = []
    peakLum = []
    peakIndex = []
    persLum = []

    startIndex = []
    endIndex = []
    startTimes = []
    endTimes = []

    fluences = []
    convexities = []
    t10 = []
    t25 = []
    t90 = []
    tBurst = []

    burstLums = []
    burstTims = []
    burstRads = []

    fitAlpha = []
    fitDecay = []

    upTurns = []

    while jj < len(tim)-200:
        # There should be a burst if  point is a maximum in +/- peakTimeWindow
        # seconds and is 10 times more luminous than 20s ago
        minTimeIndex = bn.nanargmax(tim > (tim[jj]-peakTimeBackLook))
        maxTimeIndex = bn.nanargmax(tim > (tim[jj]+peakTimeForwardLook))

        if maxTimeIndex == 0:
            break  # near end of file

        nbhd = lum[minTimeIndex:maxTimeIndex+1]
        tnbhd = tim[minTimeIndex:maxTimeIndex+1]

        endBurst = False
        if (lum[jj] > minFlux) & (lum[jj] == max(nbhd)):
            midIndex = jj - minTimeIndex
            persLumCompare =\
                lum[bn.nanargmax(tim > (tim[jj]-beginTimeBackJump))]

            if lum[jj] > 10*persLumCompare and not isShock(tnbhd, nbhd, midIndex):  # NOQA
                # We have identified a burst now
                # conditions to screen for weird bursts
                if jj >= len(tim):
                    break
                if lum[jj] > 10*lum[jj-1] and lum[jj] > 10*lum[jj+1]:
                    break
                beginIndex = jj

                recordStartIndex = np.argmax(tim > (tim[beginIndex] -
                                             beginTimeBackJump)) - 1
                currentPersLum = lum[recordStartIndex]

                endFlux = currentPersLum + 0.02*(lum[jj]-currentPersLum)
                jj = bn.nanargmax(tim > (tim[beginIndex] + 10))
                # minimum burst duration is 10 seconds
                critGrad = 0.01*(lum[jj] - currentPersLum)/10
                upTurn = False
                while lum[jj] >= endFlux:
                    if jj < len(tim)-1:
                        jj += 1
                    else:
                        print('<seperate> Beware: peak at end of data set')
                        print('<seperate> not analysing this peak, as it is' +
                              ' incomplete')
                        print(
                            '<seperate> %s other peak(s) found' % len(peakLum))
                        jj = -1
                        endBurst = True
                        break

                    if (tim[jj] - tim[beginIndex]) > 20:
                        # on long bursts, consider ending at local minima
                        tenSecAgo = bn.nanargmax(tim > (tim[jj] - 10))
                        if (lum[jj] - lum[tenSecAgo])/10. > critGrad:
                            # Check we aren't in a persistent rise
                            # This essentially checks we can fit a decay curve
                            # But also, if the burst is crazy bright still,
                            # probably not
                            # in the upturn
                            if lum[jj] < 0.8*lum[beginIndex]:
                                # check for +ve gradient
                                upTurn = True
                                print('UpTurn has occurred at' +
                                      '{0:.2f}'.format(tim[jj]))
                                break

                if endBurst is True:
                    break

                stopIndex = jj

                currentPeakLum = lum[beginIndex]
                currentPersLum = lum[recordStartIndex]

                # store the time of the peak in structure
                peakTime.append(tim[beginIndex])
                peakIndex.append(beginIndex)
                startIndex.append(recordStartIndex)
                startTimes.append(tim[recordStartIndex])
                peakLum.append(currentPeakLum)
                persLum.append(currentPersLum)

                endIndex.append(stopIndex)
                endTimes.append(tim[stopIndex])

                burstRiseTimes = tim[recordStartIndex:beginIndex+2]
                burstRiseLums = lum[recordStartIndex:beginIndex+2]

                t10.append(findAtRiseFrac(burstRiseTimes, burstRiseLums,
                           currentPeakLum, currentPersLum, .10) - peakTime[-1])
                t25.append(findAtRiseFrac(burstRiseTimes, burstRiseLums,
                           currentPeakLum, currentPersLum, .25) - peakTime[-1])
                t90.append(findAtRiseFrac(burstRiseTimes, burstRiseLums,
                           currentPeakLum, currentPersLum, .90) - peakTime[-1])

                burstLum = np.array(lum[recordStartIndex:stopIndex])
                burstTim = np.array(tim[recordStartIndex:stopIndex] -
                                    tim[beginIndex])
                burstRad = np.array(rad[recordStartIndex:stopIndex])

                burstLums.append(burstLum)
                burstTims.append(burstTim)
                burstRads.append(burstRad)

                upTurns.append(upTurn)

                tBurst.append(max(burstTim) - t25[-1])

                convexities.append(
                    convexity(burstRiseTimes, burstRiseLums, currentPeakLum,
                              currentPersLum))

                # Tail Fits
                # PeakTime is 0 at the moment
                pParams, pData, eParams, eData =\
                    burstFits(burstTim, burstLum, 0, mdl=None)

                fitAlpha.append(pParams['al'].value)
                fitDecay.append(eParams['tau'].value)

                # Now it is time to find fluence
                fluence = 0.
                for ii in np.arange(recordStartIndex, stopIndex):
                    fluence += 0.5*((lum[ii] + lum[ii+1] - 2*currentPersLum) *
                                    (tim[ii+1] - tim[ii]))
                fluences.append(fluence)
                n = len(peakLum)

                if n > 1:
                    tDel.append(peakTime[n-1] - peakTime[n-2])

                if n >= 100:
                    print('<separate> CAUTION: more than 100 peaks' +
                          'only analysing first 100')
                    print('<separate> Check data file, the data may be a' +
                          'little crazy')
                    print('<separate> This is model number %s' % modelID)
                    break
        jj += 1

    for ii in range(0, len(peakIndex)):
        fname = os.path.join(outputDirectory, '%i.data' % (ii,))
        saveArray = zip(burstTims[ii], burstLums[ii], burstRads[ii])
        headString = 'time luminosity radius'
        np.savetxt(fname, saveArray, delimiter=' ', newline='\n',
                   header=headString)
        # now we have arrays of tdels, fluences, and peak times
        # If there are peaks, then return all this data as an output

    if peakIndex != []:
        # get the tau array - the ratio of fluence to peak flux
        taus = [fl/p for (fl, p) in zip(fluences, peakLum)]
        # we can now save this in a dictionary
        return {'burstID': modelID,  # burst model number
                'num': len(peakLum),  # number of bursts
                'pTimes': peakTime,  # time of peaks
                'bstart': startTimes,  # time of recording start
                'bend': endTimes,  # end of burst
                'peakLum': peakLum,  # peak luminosity
                'persLum': persLum,  # persistent emission
                'tdel': tDel,  # tdel value
                'tau': taus,  # tau value
                'fluen': fluences,  # fluence value
                'lums': burstLums,  # luminosities
                'tims': burstTims,  # times
                'rads': burstRads,  # rads
                'conv': convexities,  # convexities for each burst
                'length': tBurst,  # duration of bursts
                't10': t10,  # time at 10% burst rise
                't25': t25,  # time at 25% burst rise
                't90': t90,  # time at 90% burst rise
                'upTurn': True in upTurns,  # is there an upTurn at burst end
                'fitAlpha': fitAlpha,  # Plaw Alpha fir values
                'fitDecay': fitDecay,  # exp tail decay timescale
                'endBurst': endBurst}  # Burst continues to end of train

    # but if there aren't any peaks
    else:
        return {'burstID': modelID,     # burst model number
                'num': 0.0,             # number of bursts
                'endBurst': endBurst}   # burst at end


class ModelAnalysis:
    """Analyse a model run of multiple bursts

    m = ModelAnalysis(modelID, time, lum, radius)

    Example Usage:
        key_table = {...}  # Load in the key table
        ma = ModelAnalysis(modelID, burstTime, burstLum, burstRad)
        flag = ma.get_flag()
        print('model: %s; flag: %i' % (modelID, flag))
        ma.separate("separated")

        dbVals = ma.get_burst_values()
        ma.get_mean_lcv("separated")
        summVals = ma.get_mean_values(
            {k: v[index] for k, v in self.key_table.items()})

    This class takes a model id code, and matching time, luminosity and radius
    arrays, and separates them into individal bursts, as well as producing a
    mean lightcurve and mean burst parameters.

    Input:
        args:
        - modelID: string/int ID of the model
        - time: 1-D array of times
        - lum:  1-D array of burst luminosities, specified at times in t
        - radius: 1-D array of burst radius profiles

    Returns:
        m: Class Object
        Default members of m:
            m.separated = {'num': 0,
                           'endBurst': False,
                           'burstID': modelID}
                endBurst is True if lightcurve ends mid-burst
        Member Functions:
            m.clean()
                Runs by default when constructor is called.
                Clean the light curve from duplicate times
                (time step smaller than numerical precision),
                negative and super-Eddington luminosities.
            m.separate(output_dir)
                Locate bursts and separate the individual burst light curves
                Updates m.separated with lightcurve parameters
                Lightcurves are saved in output_dir
                These are:
                    {'num','pTimes','bstart','bend','peakLum','persLum','tdel',
                    'tau','fluen','lums','tims','rads','conv','length','t10',
                    't25','t90','upTurn','fitAlpha','fitDecay','endBurst'}

                    All dictionary members are lists with 'num' members, except
                    'num' which is an integer.

                    'num': int, number of separated bursts
                    'pTimes': array of times of burst peaks
                    'bstart': array of times when burst starts
                    'bend': array of times when burst ends
                    'peakLum': array of peak luminosities for each burst
                    'persLum': array of persistent luminosities for each burst
                    'tdel': recurrence time of each burst (where possible)
                    'tau': array of tau values for each burst
                    'fluen': array of fluences of each burst
                    'lums': (NOTE 1) array of luminosity arrays
                    'tims': (NOTE 1) array of time arrays
                    'rads': (NOTE 1) array of radius arrays
                    'conv': array of convexities in each burst
                    'length': time duration of each burst
                    't10': array containing the time where each burst rise hits
                           10pct of peak luminosity
                    't25': array containing the time where each burst rise hits
                           25pct of peak luminosity
                    't90': array containing the time where each burst rise hits
                           90pct of peak luminosity
                    'upTurn': flag to indicate if burst has an upturn
                    'fitAlpha': array of alpha values for each burst
                    'fitDecay': timescale for the exponential decay
                    'endBurst': flag for whether burst is interrupted by the
                                end of the file/input arrays

                    NOTE 1: The list elements for these quantities are 1-D
                            arrays. They give the time, luminosity and radius
                            for each burst
            m.get_flag(twinPeaks=False, notAnalysable=False)
                Returns the flag corresponding to the analysis quality of this
                burst, see m.get_flag.__doc__ for flags
            m.get_alpha()
                Calculates the alpha values
            m.get_burst_values()
                Get slightly nicer formatted burst parameters, not including
                the time, luminosity and radius areas
            m.get_mean_lcv(output_dir=None)
                Get the mean lightcurve, specifiying the ouptut directory
                where the mean lightcurve ought to be saved (None only returns)
                the mean lc as an array
            m.get_mean_values()
                Returns the mean values for burst paramters, see
                m.get_mean_values.__doc__ for column descriptions
    """

    def __init__(self, modelID, time, lum, radius):
        """
        Initialize analysis for given time, luminosity, and radius
        arrays.
        """
        self.modelID = modelID
        self.time = time
        self.lum = lum
        self.radius = radius

        self.clean()
        # Default result of self.separate()
        self.separated = {'num': 0,
                          'endBurst': False,
                          'burstID': modelID}

    def clean(self):
        """
        Clean the light curve from duplicate times
        (time step smaller than numerical precision),
        negative and super-Eddington luminosities.
        """
        # Remove duplicate times and where lum <= 0;
        # after removal we keep the first time value,
        # rather than averaging as the values only
        # show varation at the 6th sig fig.
        ok = ((self.time - np.roll(self.time, 1)) != 0) & (self.lum > 0)
        self.time = self.time[ok]
        self.lum = self.lum[ok]
        self.radius = self.radius[ok]

        # If the Eddington luminosity is exceeded by convection effects
        # then smooth the train, to get a train more representative of
        # observations
        # use 0.125s as that is commensurate with common RXTE timebins
        # Only problem is sometimes the entire rise can happen in ~0.5s
        max_lum = np.max(self.lum)
        if max_lum > 1e39:
            logging.warn('%s exceeds Eddington, with peak lum at %e' % (self.modelID, max_lum))  # NOQA
            self.time, self.lum, self.radius = smooth(self.time,
                                                      self.lum,
                                                      self.radius,
                                                      0.125)
            self.superEddington = True
        else:
            self.superEddington = False

    def separate(self, output_dir):
        """
        Locate bursts and separate the individual burst light curves
        """
        x = separate(self.time, self.lum, self.radius, self.modelID,
                     output_dir)
        self.separated = x
        return x

    def get_flag(self, twin_peaks=False, not_analysable=False):
        """Get Analysis flags

        ModelAnalysis.get_flag(twin_peaks=False, not_analysable=False)

        Flags are used to indicate where the model may be questionable.
        These are done in base 2. The flags are:

        0:    No analysis issues
        1:    Burst at end of file (last burst not analysed)
        2:    Shocks occur that cause luminosity to exceed L>10**39 erg s^-1
        4:    Bursts have been cut at a local minimum rather than by luminosity
        8:    Bursts in this train are twin peaked, convexity should not really
              be considered for these models. Set manually
        16: Rapid bursts with recurrence time less than 100s,
            This may indicate some bursts are missed, or the observations
            include multiple bursts. Often these missed bursts are low
            intensity bursts.
        32: Burst not conducive to analysis

        x: result from separate()
        twin_peaks: whether to set flag 8
        not_analysable: whether to set flag 32
        """
        x = self.separated
        flag = int(0)
        if x['endBurst']:
            flag += 2**0  # flag 01
        if self.superEddington:
            flag += 2**1  # flag 02
        if 'upTurn' in x and x['upTurn']:
            flag += 2**2  # flag 04
        if twin_peaks:
            flag += 2**3  # flag 08
        if x['num'] > 1:
            if (np.asarray(x['tdel'][1:]) < 100).any():
                flag += 2**4  # flag 16
        if not_analysable:
            flag += 2**5  # flag 32

        self.flag = flag
        return flag

    def get_alpha(self, accretion_lum):
        """
        Return the alpha parameter for each burst
        accretion_lum: accretion luminosity in erg/s
        (can be a numpy array with values for all bursts)
        """
        x = self.separated
        alphas = [(accretion_lum + ps)*dt/fl for (ps, dt, fl)
                  in zip(x['persLum'], x['tdel'], x['fluen'])]
        x['alpha'] = alphas
        return alphas

    def get_burst_values(self):
        """
        Return the parameter values of all bursts
        """
        x = self.separated
        dbVals = []
        for i in xrange(0, x['num']):
            dbVals.append((x['burstID'],
                           i,
                           x['bstart'][i],
                           x['length'][i],
                           x['fluen'][i],
                           x['peakLum'][i],
                           x['persLum'][i],
                           x['tau'][i],
                           x['tdel'][i],
                           x['conv'][i],
                           x['t10'][i],
                           x['t25'][i],
                           x['t90'][i],
                           x['fitAlpha'][i],
                           x['fitDecay'][i],
                           x['alpha'][i]))
        return dbVals

    def get_mean_lcv(self, output_dir=None):
        """
        Calculate the mean burst light curve, and save it to 'mean.data' in
        output_dir (if output_dir is specfied)

        mean_lc = ModelAnalysis.get_mean_lcv(output_dir=None)
        returns:
            mean_lc: 2-D numpy array, columns are:
                time, luminosity, u(luminosity), radius, u(radius)
        """
        x = self.separated
        burstTime = self.time
        burstLum = self.lum
        burstRad = self.radius

        if x['num'] > 0:  # if there are bursts
            fullBTime = [a[-1] - a[0] for a in x['tims']]
            longestBurstTime = max(fullBTime)
            allTims = []
            allLums = []
            allRads = []
            for (t, b) in zip(x['tims'], x['bstart']):
                sInd = bn.nanargmax(burstTime >= (b))
                fInd = bn.nanargmax(burstTime >= (b+longestBurstTime))
                if fInd == 0:
                    fInd = len(burstTime)
                allTims.append(burstTime[sInd:fInd] - b + t[0])
                allLums.append(burstLum[sInd:fInd])
                allRads.append(burstRad[sInd:fInd])

            mt, ml, mr, mdl, mdr = avgParams(allTims, allLums, allRads)
            saveArray = zip(mt, ml, mdl, mr, mdr)
            if output_dir is not None:
                fname = os.path.join(output_dir, self.modelID, 'mean.data')
                print('writing burst to %s' % fname)
                headString = 'time luminosity u_luminosity radius u_radius'
                np.savetxt(fname, saveArray, delimiter=' ', newline='\n',
                           header=headString)
            return saveArray

    def get_mean_values(self, key_table):
        """
        Get mean burst parameters
        Usage:
            summ_vals: ModelAnalysis.get_mean_values(key_table)
        args:
            key_table: keys for the specific model that is being analysed

        returns:
            summ_vals: list of mean values, values are:
                idx  value
                00. modelID
                01. number of bursts
                02. accretion rate (from table)
                03. metallicity (from table)
                04. hydrogen fraction (from table)
                05. accretion luminosity (from table)
                06. pulse column (from table)
                07. number of cycles (from table)
                08. burstLength,
                09. uBurstLength,
                10. peakLum,
                11. uPeakLum,
                12. persLum,
                13. uPersLum,
                14. fluence,
                15. uFluence,
                16. tau,
                17. uTau,
                18. tDel,
                19. uTDel,
                20. conv,
                21. uConv,
                22. r1090,
                23. uR1090,
                24. r2590,
                25. uR2590,
                26. singAlpha,
                27. uSingAlpha,
                28. singDecay,
                29. uSingDecay,
                30. alpha,
                31. uAlpha,
                32. self.flag,
                33. Q_b
        """
        x = self.separated

        if x['num'] > 0:  # if there are bursts
            if x['num'] >= 3:
                # Ignore the first burst in train if there are over 3 bursts
                for value in x.itervalues():
                    if type(value) == list:
                        value.remove(value[0])

            peakLum = np.mean(x['peakLum'])
            uPeakLum = np.std(x['peakLum'])

            persLum = np.mean(x['persLum'])
            uPersLum = np.std(x['persLum'])

            burstLength = np.mean(x['length'])
            uBurstLength = np.std(x['length'])

            fluence = np.mean(x['fluen'])
            uFluence = np.std(x['fluen'])

            tau = np.mean(x['tau'])
            uTau = np.std(x['tau'])

            conv = np.mean(x['conv'])
            uConv = np.std(x['conv'])

            singAlpha = np.mean(x['fitAlpha'])
            uSingAlpha = np.std(x['fitAlpha'])

            singDecay = np.mean(x['fitDecay'])
            uSingDecay = np.std(x['fitDecay'])

            # rise time 10%-90%
            t10t90 = np.array(x['t90']) - np.array(x['t10'])
            r1090 = np.mean(t10t90)
            uR1090 = np.std(t10t90)

            # rise time 25%-90%
            t25t90 = np.array(x['t90']) - np.array(x['t25'])
            r2590 = np.mean(t25t90)
            uR2590 = np.std(t25t90)

            if x['num'] >= 2:
                tDel = np.mean(x['tdel'][1:])
                uTDel = np.std(x['tdel'][1:])
                alpha = np.mean(x['alpha'][1:])
                uAlpha = np.std(x['alpha'][1:])
            else:
                tDel = 0.
                uTDel = 0.
                alpha = 0.
                uAlpha = 0.
        else:  # No bursts case: set all the things to zero
            burstLength = 0
            uBurstLength = 0

            peakLum = 0
            uPeakLum = 0

            persLum = 0
            uPersLum = 0

            fluence = 0
            uFluence = 0

            tau = 0
            uTau = 0

            tDel = 0
            uTDel = 0

            conv = 0
            uConv = 0

            r1090 = 0
            uR1090 = 0

            r2590 = 0
            uR2590 = 0

            alpha = 0
            uAlpha = 0

            singDecay = 0
            uSingDecay = 0

            singAlpha = 0
            uSingAlpha = 0

        summVals = (self.modelID,
                    x['num'],
                    key_table['acc'],
                    key_table['z'],
                    key_table['h'],
                    key_table['lAcc'],
                    key_table['pul'],
                    key_table['cyc'],
                    burstLength,
                    uBurstLength,
                    peakLum,
                    uPeakLum,
                    persLum,
                    uPersLum,
                    fluence,
                    uFluence,
                    tau,
                    uTau,
                    tDel,
                    uTDel,
                    conv,
                    uConv,
                    r1090,
                    uR1090,
                    r2590,
                    uR2590,
                    singAlpha,
                    uSingAlpha,
                    singDecay,
                    uSingDecay,
                    alpha,
                    uAlpha,
                    self.flag,
                    key_table['qb'])
        return summVals
