###############################################################################
'''
model_analysis.py
Author: Nathanael Lampe, Monash University
Date Created: 19/7/2013 (Anniversary of the moon landing, woohoo)

Purpose:
To recode old IDL routines that took KEPLER models and produced mean
lightcurves and burst parameters.

Changelog:

'''
###############################################################################
from __future__ import division, unicode_literals, print_function
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chisqprob
from copy import deepcopy
from lmfit import minimize, Parameters, report_fit
from itertools import izip
import bottleneck as bn
import os
import pdb

np.seterr(divide='raise', over='warn', under='warn', invalid='raise')

###############################################################################
###############################################################################
#####                                                                     #####
#####                        ANALYSIS SUBROUTINES                         #####
#####                                                                     #####
###############################################################################
###############################################################################

#####~~~~~~~~~~~~~~~~~~~~~~~~~~FITTING BURST TAILS~~~~~~~~~~~~~~~~~~~~~~~######

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
              'exp1':  fit a single exponential [1]
              'exp2':  fit a double exponential (not implemented)

  returns:
  -----
  params:  lmfit.Parameters object with fit parameters
  fitData: dictionary with fit statistics

  =============================================================================
  Notes:
  
  1. The lmfit.Parameters class object specifies parameter values and bounds.
       The parameters it needs to specify vary depending on the fitting method,
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

#opening assertions
  assert len(time) == len(lumin), 'time and lumin should be same length'
  assert np.ndim(time) == 1, 'time & lumin should be 1-d'
  assert method in ['power','exp1','exp2'], 'method kwarg invalid'

#copy arrays
  time = deepcopy(np.asarray(time))
  lumin = deepcopy(np.asarray(lumin))
  dlumin = deepcopy(np.asarray(dlumin))
  

  def exp1(t,p):
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
    1-D array of burst luminosities as predicted by a single tail exponential
      with parameters set by params in p

    '''
    F0 = p['F0'].value
    t0 = p['t0'].value
    tau = p['tau'].value
    Fp = p['Fp'].value

    return F0*np.exp(-(t-t0)/tau) + Fp


  def power(t,p):
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
    return F0*((t-ts)/(t0-ts))**(-1*al) + Fp
  
  msg = 'Two tail exponential not available yet'

  if method == 'power':  f = power
  elif method == 'exp1': f = exp1
  elif method == 'exp2': raise NotImplementedError(msg)
  else: raise RuntimeError('Method kwarg incorrect')
  
  def minFunc(p, t, l, dl, f):
    '''
    Minimisation Function
    '''
    lm = f(t, p)
    csq = ((l-lm)/dl)
    return csq



  result = minimize(minFunc, params, args = (time, lumin, dlumin, f),
    method='leastsq')
  
  fitData = { 'nfev'         : result.nfev,         #number of function
                                                    #evaluations
              'success'      : result.success,      #boolean (True/False) 
                                                    #for whether fit succeeded
              'errorbars'    : result.errorbars,    #boolean (True/False) for
                                                    #whether uncertainties were
                                                    #estimated.
              'message'      : result.message,      #message about fit success.
              'ier'          : result.ier,          #integer error value from 
                                                    #scipy.optimize.leastsq
              'lmdif_message': result.lmdif_message,#message from 
                                                    #scipy.optimize.leastsq
              'nvarys'       : result.nvarys,       #number of variables in fit  
              'ndata'        : result.ndata,        #number of data points:  
              'nfree'        : result.nfree,        #degrees of freedom in fit:  
              'residual'     : result.residual,     #residual array 
                                                    #(return of func():  
              'chisqr'       : result.chisqr,       #chi-square: 
              'redchi'       : result.redchi}       #reduced chi-square: 
            
  return params, fitData

#####~~~~~~~~~~~~~~~~~~~~~~~~ Fitting routine ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####

def burstFits(mt,ml,pt,mdl=None):
  '''
  Conduct a least squares minimisation of a power law and single tail exponential
  on a thermonuclear burst tail
  
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
  pData:   dictionary with fit statistics for pLaw fit
  eParams: lmfit.Parameters object with exponential fit final parameters
  eData:   dictionary with fit statistics for exp law fit
  In both cases, pp['rcs']

  =============================================================================
  Notes:
  Conducts fits to burst tails following In't Zand et al, A&A 562 A16 (2014),
  except fit from 0.90Fpeak to 0.10Fpeak, ratehr than adaptively choose where to
  start fitting.
  '''
  
  peakTime = bn.nanargmax(mt>pt)
  peakLum = ml[peakTime]
  persLum = ml[0]/peakLum #(scaled persLum)

  #rescaling, by 1/peakLum, and set equal weights if mdl == None
  if mdl == None:
    t = mt[peakTime:]
    l = ml[peakTime:]/peakLum
    dl = np.ones(len(t))
  else:
    ok = np.nonzero( (mdl>1e35) & (mt>0))[0] #remove artificially low uncertainties
    t = mt[ok]
    l = ml[ok]/peakLum
    dl = mdl[ok]/peakLum
 
  #find t90, t10
  
  n = len(l)
  t10Ind = n - bn.nanargmax(l[::-1] > 0.1) #last occurence crossing
                                           #0.1Lpeak (note >)
  t90Ind = n - bn.nanargmax(l[::-1] >= 0.9) #find last point where we 
                                            #have 0.9Lpeak (note >=) 
  t0 = t[t90Ind]
  tFit = t[t90Ind:t10Ind]
  lFit = l[t90Ind:t10Ind]
  dlFit = dl[t90Ind:t10Ind]

  p = Parameters()

  p.add('F0', value = 1., vary = True)
  p.add('t0', value = t0, vary = False)
  p.add('ts', value = -1, vary = True, min = -10, max = t0)
  p.add('al', value = 1.4, vary = True, min = 0)
  p.add('Fp', value = persLum, vary = True, min = 0., max = 1.)


  pParams, pData = fitTail(tFit, lFit, dlFit, p, method = 'power')

  p = Parameters()

  p.add('F0', value = 1., vary = True)
  p.add('t0', value = t0, vary = False)
  p.add('tau', value = 10, vary = True, min = 0)
  p.add('Fp', value = persLum, vary = True, min = 0., max =1.)


  eParams, eData = fitTail(tFit, lFit, dlFit, p, method = 'exp1')
  '''
  OLD METHOD (Following In't Zand)
  #Power Law
  ts1 = 3
  ts2 = ts1+1

  sInd1 = bn.nanargmax(t>ts1) #start 3s after peak
  ts1 = t[sInd1]
  sInd2 = bn.nanargmax(t>ts2)
  ts2 = t[sInd2]
  endIndex = bn.nanargmax(l<10*persLum) #end when lum gets to
                                            #same order as persistent
  if endIndex == 0: endIndex = len(l) - 1 #in case it never gets there


  t1 = t[sInd1:endIndex]
  l1 = l[sInd1:endIndex]
  dl1 = dl[sInd1:endIndex]

  t2 = t[sInd2:endIndex]
  l2 = l[sInd2:endIndex]
  dl2 = dl[sInd2:endIndex]

  p = Parameters()

  p.add('F0', value = 1., vary = True)
  p.add('t0', value = ts1, vary = False)
  p.add('ts', value = -1, vary = True, min = -10, max = ts1)
  p.add('al', value = 1.4, vary = True, min = 0)
  p.add('Fp', value = persLum, vary = False, min = 0., max = 1.)


  p1, data1 = fitTail(t1, l1, dl1, p, method = 'power')
  p['ts'].max = 0.999*ts2       
  p['t0'].value = ts2
  p2, data2 = fitTail(t2, l2, dl2, p, method = 'power')

  while data2['redchi'] < data1['redchi'] or data2['redchi'] >= 5.:  

    ts2 += 1
    if ts2 - max(t) < 5.0: break #fit to at least 5s
    data1 = data2
    p1 = p2

    sInd2 = bn.nanargmax(t>ts2)
    ts2 = t[sInd2]
    t2 = t[sInd2:endIndex]
    if len(t2)<6: break
    l2 = l[sInd2:endIndex]
    dl2 = dl[sInd2:endIndex]
    p['ts'].max = 0.999*ts2
    p['t0'].value = ts2
    p2, data2 = fitTail(t2, l2, dl2, p, method = 'power')

  pParams=p1
  pData = data1
  
  #~~~ Single Tail Exponential ~~~#
        
  ts1 = 3
  ts2 = ts1+1

  sInd1 = bn.nanargmax(t>ts1) #start 3s after peak
  ts1 = t[sInd1]
  sInd2 = bn.nanargmax(t>ts2)
  ts2 = t[sInd2]
  endIndex = bn.nanargmax(l<10*persLum) #end when lum gets to
                                            #same order as persistent
  if endIndex == 0: endIndex = len(l) - 1 #in case it never gets there


  t1 = t[sInd1:endIndex]
  l1 = l[sInd1:endIndex]
  dl1 = dl[sInd1:endIndex]

  t2 = t[sInd2:endIndex]
  l2 = l[sInd2:endIndex]
  dl2 = dl[sInd2:endIndex]

  p = Parameters()

  p.add('F0', value = 1., vary = True)
  p.add('t0', value = ts1, vary = False)
  p.add('tau', value = 10, vary = True, min = 0)
  p.add('Fp', value = persLum, vary = False, min = 0., max =1.)


  p1, data1 = fitTail(t1, l1, dl1, p, method = 'exp1')
  
  p['t0'].value = ts2
  p2, data2 = fitTail(t2, l2, dl2, p, method = 'exp1')

  while data2['redchi'] < data1['redchi'] or data2['redchi'] >= 5.:

    ts2 += 1
    if ts2 - max(t) < 5.0: break #fit to at least 5s
    data1 = data2
    p1 = p2

    sInd2 = bn.nanargmax(t>ts2)
    ts2 = t[sInd2]
    t2 = t[sInd2:endIndex]
    if len(t2)<6: break
    l2 = l[sInd2:endIndex]
    dl2 = dl[sInd2:endIndex]

    p['t0'].value = ts2
    p2, data2 = fitTail(t2, l2, dl2, p, method = 'exp1')
    
  eParams = p1
  eData = data1
  '''

  #rescale back
  pParams['F0'].value  *= peakLum
  pParams['F0'].stderr *= peakLum
  pParams['Fp'].value  *= peakLum
  pParams['Fp'].stderr *= peakLum

  eParams['F0'].value  *= peakLum
  eParams['F0'].stderr *= peakLum
  eParams['Fp'].value  *= peakLum
  eParams['Fp'].stderr *= peakLum
  return pParams,pData,eParams,eData


#####~~~~~~~~~~~~~~~~~~~~~~~~~RISE TIMES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####

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

  #find candidates where we cross the required Lumfrac
  greater = lumin > fracLum
  cross = greater - np.roll(greater,1)
  cands = np.nonzero(cross[1:])[0] + 1 #candidates to cross fracLum

  avgBurstGrad = (lumin[-1]-lumin[0])/(time[-1]-time[0])
  
  grads = (lumin[cands]-lumin[cands-1])/(time[cands]-time[cands-1])
  #exclude -ve gradients (added apr-03-2014)
  for ii in xrange(len(grads)):
    if grads[ii] < 0:
      grads[ii] = np.Inf


  #calculate gradient by back-step FD, rather than central diff
  candGrads = zip(cands, grads)
  candGrads.sort(key = lambda x: x[1]) #sort by gradient 

  best = candGrads[0][0] #Most likely candidate has lowest gradient

#Now interpolate for more precision
  interpTimes = np.linspace(time[best-1],time[best+1],2**6)
  lumInterp = interp1d(time, lumin, kind='linear', copy=True)
  interpLums = lumInterp(interpTimes)
  interpBest = bn.nanargmax(interpLums>fracLum)
  
  return interpTimes[interpBest]


#####~~~~~~~~~~~~~~~~~~~~~~~~~SMOOTHING LIGHTCURVES~~~~~~~~~~~~~~~~~~~~~~~#####

def smooth(t,f,r,tmin):
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

  A routine to take a lightcurve and return a new lightcurve that averages over
  variations that last less than tmin.
  Specifically designed to be used on lightcurves that show large shocks e.g.
  model a6. In these cases, a number of millisecond shocks cause large 
  increases in stellar luminosity due to a convective layer reaching the 
  stellar surface. These are too small to typically be observed, and boggle
  the analysis routine that tries to identify peak luminosities
  '''
  #some simple asseritons
  assert len(t) == len(r) and len(t) == len(f), '''t,r,f should be the same
    shape'''

  assert len(np.shape(t)) == 1, '''t should be 1-D'''
  #set up lists, loops
  ii=0
  newt=[]
  newf=[]
  newr=[]
  dt = t - np.roll(t,1)
  n = len(t)

  while ii < n:
    if dt[ii]>tmin:
      #case 1: bigger timestep than tmin
      newt.append(t[ii])
      newf.append(f[ii])
      newr.append(r[ii])
      ii+=1
    elif dt[ii] < 0:
      #case 2: first timestep will have dt<0
      newt.append(t[ii])
      newf.append(f[ii])
      newr.append(r[ii])
      ii+=1
    else:
      #case 3: smaller timesteps than tmin
      tStart = ii
      tStop = max(tStart, np.argmax(t>(t[ii]+tmin)))
      if (tStop != tStart) and tStop != n:
        #3a: not at the EOF
        newt.append(t[ii]+tmin/2.)
        avgf = sum(f[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])
        avgr = sum(r[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])
        newf.append(avgf)
        newr.append(avgr)
        ii=tStop
      else:
        #3b found the EOF, average to EOF
        tStop = n-1
        newt.append( (t[tStart]+t[tStop])/2.)
        avgf = sum(f[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])
        avgr = sum(r[tStart:tStop]*dt[tStart+1:tStop+1])/(t[tStop]-t[tStart])
        newf.append(avgf)
        newr.append(avgr)
        ii = n
  return np.array(newt),np.array(newf),np.array(newr)

#####~~~~~~~~~~~~~~~~~~~~~~~~~~Shock Checking~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####

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
  #some assertions
  assert len(np.shape(times)) == 1, ''' times should be 1-d'''
  assert index == np.int_(index), '''index should be type int'''
  assert np.shape(times) == np.shape(fluxes), '''times and fluxes should have
    the same shape'''
  

  #shocks do not change the luminosity much either side of the shock, but 
  #are vastly different to the shock
  #get indices for +/- 0.05s from shock

  justPrev = bn.nanargmax(times > (times[index] - 0.02 ))
  if justPrev == index: justPrev -= 1
  justAfter = bn.nanargmax(times > (times[index] + 0.02 ))
  if justAfter == index: justAfter += 1

  if ( (fluxes[justPrev]/fluxes[index] > 0.8) or 
    (fluxes[justAfter]/fluxes[index] > 0.8) ):
    #we have no shock, as the values either side of this point suggest
    #a flat top, or is a step
    if fluxes[index+1] < 0.95*fluxes[index]:
      #extra little routine to pick up shocks near the burst peak
      shock = True
    else:
      shock = False
  else:
    shock = True

  return shock

#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Simpson's Rule~~~~~~~~~~~~~~~~~~~~~~~~~#####

def simprule(a,b,c,dt):
  '''
  simprule(a,b,c,dt)
  evaluate the area underneath three points by Simpson's Rule
  the three points ought to be equally spaced, with b being the midpoint
  the inputs a,b,and c are f(start), f(mid) and f(end) respectively,
  dt is the spacing
  
  Thus, Simpson's rule becomes here: area=([c-a]/6)(f(a)+4f(b)+f(c))
  '''
  return (a+4.0*b+c)*dt/6.0

#####~~~~~~~~~~~~~~~~~~~~~~~Check Local Maximum~~~~~~~~~~~~~~~~~~~~~~~~~~~#####

def isMaximum(a,b,c):
  '''
  Small routine to check if point b is a local maximum
  '''
  if b>a and b>c: return True
  else: return False

###############################################################################
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~Convexity~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
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
  peakLum -= persLum #correct for persistent luminosity

  n = len(lumin)
  
  #sampleRate = 1/min((time - np.roll(time,1))[1:])
  #ex = 10.
  #while 2**ex < sampleRate and ex <17: ex += 1
  ##so I don't kill memory, restrict N to 4M
  #newN = 2.**(ex+5.) #ensure finer spacing in interpolation (+5 accounts for
                   #20 s window
  #newN = 2.**22.

  #interpolate to find convexity on a fine grid
  #lumInterp = interp1d(time, lumin, kind='linear', copy=True)
  #time = np.linspace(time[0], time[n-1], newN)
  #lumin = lumInterp(time)
  #n = newN
  #screw interpolation, lets just do some linear equations
  #after all, I do a trapezoidal integration, so we don't actually
  #lose or gain any information
  gt10 = lumin > 0.1*peakLum
  ten = (n) - bn.nanargmin(gt10[::-1])
  gt90 = lumin > 0.9*peakLum
  ninety = (n) - bn.nanargmin(gt90[::-1])
#  print(ten, ninety)

  #for 10%
  fa = 0.1*peakLum
  m = (lumin[ten] - lumin[ten-1])/(time[ten] - time[ten-1])
  ta = time[ten-1] + (fa - lumin[ten-1])/m

  #for 90%
  fb = 0.9*peakLum
  m = (lumin[ninety] - lumin[ninety-1])/(time[ninety] - time[ninety-1])
  tb = time[ninety-1] + (fb - lumin[ninety-1])/m

#  ten = np.argmax(lumin > 0.1*peakLum)    #L=10% burst height
#  ninety = np.argmax(lumin > 0.9*peakLum) #L=90% burst height
  


  t = np.concatenate(([ta], time[ten:ninety], [tb]))
  l = np.concatenate(([fa], lumin[ten:ninety], [fb])) 
  #relevant time and lum during rise
  
  lRange = max(l) - min(l)
  if lRange !=0:
#    lCrit = l[0]-.05*lRange
#    #check to see if we are starting at a kink
#    if min(l) <= lCrit: #a kink is identified if luminosity drops to -5%
#                      #of its scaled range
#      #rescale to ignore kink                  
#      lMinIndex = np.argmax(l <= lCrit)
#      ten = np.argmax(l[lMinIndex:] > 0.1*peakLum) + lMinIndex
#      t = time[ten:ninety]
#      l = lumin[ten:ninety]
    #/kinkiness check

    #normalise t, l to be between 0 and 10
    t -= t[0]
    l -= l[0] #0.1*peakLum #l[0]
    lastl = l[-1]
    lastt = t[-1]
    t *= (10./lastt)
    l *= (10./lastl)

    #/normalisation
#    pdb.set_trace() 
    #calculate convexity
    c = 0
    for ii in xrange(1,len(t)):
      c += 0.5*( (l[ii] + l[ii-1]) - (t[ii] + t[ii-1]) )*(t[ii] - t[ii-1])
      #trapezoidal integration implemented 14-11-2013
    return c
  else:
    return 0

###############################################################################
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LC Averaging~~~~~~~~~~~~~~~~~~~~~~~~~~~~#####
###############################################################################

####~~~~~~~~~~~~~~~~~~~~~~~~~~Average to key burst~~~~~~~~~~~~~~~~~~~~~~~~#####

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
  elif n==2:
    lens = map(max, times)
    if lens[1]>lens[0]:
      #Average using longer array
      times.reverse()
      lums.reverse()
      rads.reverse()
    t = times[0]
    #l = 0.5*lums[0]
    #r = 0.5*rads[0]
    interpLum = interp1d(times[1],lums[1], fill_value = np.nan,
      bounds_error=False)
    interpRad = interp1d(times[1],rads[1], fill_value = np.nan,
      bounds_error=False)
    l = np.nanmean([lums[0], interpLum(t)], axis = 0) # 0.5*interpLum(t)
    r = np.nanmean([rads[0], interpRad(t)], axis = 0) # 0.5*interpRad(t)
    
    dl = np.zeros(len(t))
    dr = np.zeros(len(t))
    
  elif n>=3:
    #exclude first burst, interpolate along 2, then average
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
        interpLum = interp1d(times[ii],lums[ii], fill_value = np.nan, 
          bounds_error=False)
        interpRad = interp1d(times[ii],rads[ii], fill_value = np.nan, 
          bounds_error=False)
        interpolatedLums.append(interpLum(t))
        interpolatedRads.append(interpRad(t))

    interpolatedLums = np.asarray(interpolatedLums, dtype=np.float64)
    interpolatedRads = np.asarray(interpolatedRads, dtype=np.float64)
    #can we do some outlier analysis here to remove shocks in the rise?
    #a question for later analyses.
    
    l = np.nanmean(interpolatedLums, axis=0)
    r = np.nanmean(interpolatedRads, axis=0)
   
    dl = np.nanstd(interpolatedLums, axis=0, ddof = 0)
    dr = np.nanstd(interpolatedRads, axis=0, ddof = 0)
    #now remove any sneaky NaN's that have gotten through
    #shouldn't have to do this, but apparently I do.
    #NaN's may arise from every column in interpolatedLums being NaN, but we
    #shouldn't interpolate here

    bad_dl = np.isnan(dl)
    bad_dr = np.isnan(dr)
    ok = np.logical_not( np.logical_or( bad_dl, bad_dr ) )
    t = t[ok]
    l = l[ok]
    r = r[ok]
    dl = dl[ok]
    dr = dr[ok]


    #There will still be spots where dl=0 (std of one value) but now there are
    #no NaN's!!
  return t, l, r, dl, dr
  
####~~~~~~~~~~~~~~~~~~~~~~Average to fixed interval~~~~~~~~~~~~~~~~~~~~~~######

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
  =============================================================================

  Notes:
  This routine considers that observations integrate across 0.125s intervals
  in x-ray detectors.
  Hence, rather than binning to the model resolution, we consider integrating
  the observed flux in fixed width bins. This defaults to 0.125s, as this is 
  the finest bin-width commonly encountered
  =============================================================================
  Changelog:
  Created 25-11-2013 from avgParams

                
  '''

  def interp(times,lums,rads,interval):
    '''
    Call this routine in multi-burst averaging
    '''
    #make new time array
    minTime = min([t[0] for t in times])
    maxTime = max([t[-1] for t in times])

    posTime = np.arange(0, maxTime, interval)
    negTime = np.arange(-1*interval, minTime, -1*interval)
    newTime = np.concatenate( (negTime[::-1],posTime) )
    
    #now we integrate across each bin
    newL = []
    newR = []
      
    for (t,l,r) in zip(times,lums,rads):

      newl = [] #define lists for new burst
      newr = []
      for tBin in newTime:
        #tbin is the bin midpoint for integration
        mint = tBin - interval/2
        maxt = tBin + interval/2
        goodIndex = np.nonzero(np.logical_and(t>mint, t<maxt))[0]
    
        if len(goodIndex) > 1:
          mni = goodIndex[0]
          mxi = goodIndex[-1]
          try:
            #if the points all exist, integrate
            #startpoints
            l0 = l[mni-1] + (l[mni]-l[mni-1])/(t[mni]-t[mni-1])*(mint-t[mni-1])
            r0 = r[mni-1] + (r[mni]-r[mni-1])/(t[mni]-t[mni-1])*(mint-t[mni-1])
            #first trapezoid
            ll = 0.5*(l[mni]+l0)*(t[mni]-mint)
            rr = 0.5*(r[mni]+r0)*(t[mni]-mint)
            #middle trapezoids
            for ind in goodIndex[1:]:
              ll += 0.5*(l[ind]+l[ind-1])*(t[ind]-t[ind-1])
              rr += 0.5*(r[ind]+r[ind-1])*(t[ind]-t[ind-1])
            #endpoints
            lf = l[mxi] + (l[mxi+1]-l[mxi])/(t[mxi+1]-t[mxi])*(maxt-t[mxi])
            rf = r[mxi] + (r[mxi+1]-r[mxi])/(t[mxi+1]-t[mxi])*(maxt-t[mxi])
            #last trapezoid
            ll += 0.5*(l[mxi+1]+lf)*(maxt-t[mxi])
            rr += 0.5*(r[mxi+1]+rf)*(maxt-t[mxi])
            #normalise
            ll /= interval
            rr /= interval
          except IndexError:
            ll = np.NaN
            rr = np.NaN
        elif len(goodIndex) in [1]:
          ll = l[goodIndex[0]]
          rr = r[goodIndex[0]]
        else:
          idx = bn.nanargmax(t>tBin)
          if idx != 0:
            ll = l[idx-1] + (l[idx]-l[idx-1])/(t[idx]-t[idx-1])*(tBin-t[idx-1])
            rr = r[idx-1] + (r[idx]-r[idx-1])/(t[idx]-t[idx-1])*(tBin-t[idx-1])
          else:
            ll = np.NaN
            rr = np.NaN
        newl.append(ll)
        newr.append(rr)
      newL.append(np.array(newl))
      newR.append(np.array(newr))
    

    #and now take averages. Hooray
    t = newTime
    l = np.nanmean(np.asarray(newL), axis=0)
    r = np.nanmean(np.asarray(newR), axis=0)
    dl = np.nanstd(np.asarray(newL), axis=0, ddof = 0)
    dr = np.nanstd(np.asarray(newR), axis=0, ddof = 0)
    
    #now remove any sneaky NaN's that have gotten through
    #shouldn't have to do this, but apparently I do.
    #NaN's may arise from every column in interpolatedLums being NaN, but we
    #shouldn't interpolate here

    bad_dl = np.isnan(dl)
    bad_dr = np.isnan(dr)
    ok = np.logical_not( np.logical_or( bad_dl, bad_dr ) )
    t = t[ok]
    l = l[ok]
    r = r[ok]
    dl = dl[ok]
    dr = dr[ok]

    #There will still be spots where dl=0 (std of one value) but now there are
    #no NaN's!!
    return t,l,r,dl,dr
    
    #~~~~

  #Commence routine
  n = len(times)

  if n == 1:
  #cannot really perform an average
    dl = np.zeros(len(times[0]))
    dr = np.zeros(len(times[0]))
    t = times[0]
    l = lums[0]
    r = rads[0]
    
  elif n==2:
  #Average second burst with first
    t,l,r,dl,dr =  interp(times,lums,rads,interval)
  
  elif n>=3:
    #average bursts from second onwards
    t,l,r,dl,dr = interp(times[1:],lums[1:],rads[1:],interval)

  return t,l,r,dl,dr
#####~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~######


###############################################################################
#####                          FIND STABLE BURNERS                        #####
###############################################################################
def findStable():
  '''
  NOTE: OBSOLETE: LOOKS RELIES ON OLD HARD-CODED DIRECTORIES
  
  Return a list of burners that have a burning transtion

  findStable()
  =============================================================================
  args
  -----
    None

  kwargs
  -----
    None

  Returns
  -----
    List
      A list of burst names 'axxx' that transition to different stable burning

  =============================================================================
  Purpose:

  We want to define which stars transition to stable burning. This can by large
  be done automatically, by looking at the starting and finishing luminosity of
  the burst train.
  
  =============================================================================
  Method:

  Look at the luminosity at the start of the train (averaged across first 10s)
  and compare this with the averaged luminosity across the last ten seconds.
  If they are significantly different, it indicates  either a burst at the 
  end of the file or a transition to stable burning.

  Check to assert there is no burst, and then we know that there is (likely)
  a transition to a different steady burning regime.
  '''
  #INITIAL DEFINITIONS
  modelDirectory = '/home/nat/xray/models-data/models/'

  #define the list of bursts to check
  
  #none existant models
  noExist = [2,228,233,235,257,258,259,260,261,331]
  noEntry = [1] #no entry in MODELS.txt

  bad = np.array(noEntry + noExist, dtype=int) 
 
  allModelIDs = np.arange(1,475,dtype=int)
  additionalModels = ['a5d'] #for models added to analysis
  goodModelIDs = (['a'+str(ii) for ii in allModelIDs if ii not in bad] + 
    additionalModels)
  
  transition = []
  for modelID in goodModelIDs:
    print(modelID)
    loadFilename = modelDirectory+'xrb'+str(modelID)+'.data'
    burstTime,burstLum,burstRad = np.loadtxt(loadFilename, skiprows=1, 
      unpack=True)
    
    n = len(burstTime)
    first10 = bn.nanargmax(burstTime>10)
    last10 = bn.nanargmax(burstTime>(max(burstTime)-10)) - 2
    dt = np.roll(burstTime, -1) - burstTime
 
    first10avg = sum(burstLum[0:first10]*dt[0:first10])/burstTime[first10]
    last10avg = sum(burstLum[last10:n-2]*dt[last10:n-2])/(burstTime[n-2] - 
      burstTime[last10])

    if last10avg>30*first10avg:
      #probably have a transition
      #check we don't have a burst
      #compare max abnd minimum lums in the last 1000s to do this
      
      bEnd = bn.nanargmax(burstTime>(max(burstTime)-1000)) - 2

      maxL = max(burstLum[bEnd:])
      minL = min(burstLum[bEnd:])

      if maxL < 10*minL:
        #Burst unklikey, lums are steady across last 1000s
        transition.append(modelID)

  return transition

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###


###############################################################################
###############################################################################
#####                                                                     #####
#####                    INDIVIDUAL BURST SEPARATION                      #####
#####                                                                     #####
###############################################################################
###############################################################################

def separate(burstTime, burstLum, burstRad, modelID, outputDirectory):
  '''
  separate(bursttime, burstflux, burstrad, modelID, outputDirectory)
  This program is designed to seperate the bursts in the models into individual
  burst files for subsequent analysis. It will take the delta t value's for each
  burst
  Individual burst files do not have the persistent luminosity subtracted
  '''
  print('SEPARATING '+str(modelID))
  beginTimeBackJump = 20 #seconds before burst to start recording
  peakTimeBackLook = 0.3 #look for the peak in a 5 second range
  peakTimeForwardLook = 5.0 #look forward 5s, and back less
  tim = np.asarray(burstTime, dtype=np.float64)
  lum = np.asarray(burstLum, dtype=np.float64)
  rad = np.asarray(burstRad, dtype=np.float64)
  minFlux = 1.e36
  '''
  DEPRECATED: NO REDSHIFTING FOR NOW
  #REDSHIFTING
  # Following Keek (2011) - arxiv:1110.2172 - We use the redshift scaling constants:
  # This is for a r=10km, 1.4Msun Netron Star
  print '<separate>: redshifting burst'
  z = 0.25835
  xi = 1.12176
  zeta = 0.206666
  phi = 1.
  
  
  tim *= (1.+z)
  lum *= (xi^2./(1.+z)**2.)
  rad *= (xi*(1.+z))

  #END REDSHIFTING
  '''

  #ANALYSIS
  
  jj=bn.nanargmax(tim > beginTimeBackJump) + 1
  startTime = np.argmax(tim>peakTimeBackLook)
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
    #There should be a burst if the point is a maximum to +/- peakTimeWindow
    #seconds and is 10 times more luminous than 20s ago
    minTimeIndex = bn.nanargmax(tim>(tim[jj]-peakTimeBackLook))
    maxTimeIndex = bn.nanargmax(tim>(tim[jj]+peakTimeForwardLook))
    
    if maxTimeIndex == 0: break #near end of file

    nbhd = lum[minTimeIndex:maxTimeIndex+1]
    tnbhd = tim[minTimeIndex:maxTimeIndex+1]
    
    endBurst = False
    if (lum[jj] > minFlux) & (lum[jj]==max(nbhd)):

      midIndex = jj - minTimeIndex
      persLumCompare = lum[bn.nanargmax(tim>(tim[jj]-beginTimeBackJump))]

      if lum[jj]>10*persLumCompare and not isShock(tnbhd, nbhd, midIndex):
        #We have identified a burst now

        #conditions to screen for weird bursts
        if jj >= len(tim): break
        if lum[jj] > 10*lum[jj-1] and lum[jj] > 10*lum[jj+1]:break
        #/screening
        beginIndex = jj
        
        
        recordStartIndex = np.argmax(tim > (tim[beginIndex] - 
          beginTimeBackJump)) - 1
        currentPersLum = lum[recordStartIndex]
        
        endFlux = currentPersLum + 0.02*(lum[jj]-currentPersLum)
        jj = bn.nanargmax( tim > (tim[beginIndex] + 10) )
        #minimum burst duration is 10 seconds
        
        critGrad = 0.01*(lum[jj] - currentPersLum)/10
        upTurn = False
        while lum[jj] >= endFlux:
          if jj < len(tim)-1:
              jj+=1
          else:
            print('<seperate> Beware: peak at end of this data set')
            print('<seperate> not analysing this peak, as it is incomplete')
            print('<seperate> %s other peak(s) found' % len(peakLum))
            jj = -1
            endBurst = True
            break
            
          if (tim[jj] - tim[beginIndex]) > 20:
            #on long bursts, consider ending at local minima
            tenSecAgo = bn.nanargmax( tim > ( tim[jj] - 10 ) )
            if (lum[jj] - lum[tenSecAgo])/10. > critGrad:
              #Check we aren't in a persistent rise
              #This essentially checks we can fit a decay curve
              #But also, if the burst is crazy bright still, probably not
              #in the upturn
              if lum[jj] < 0.8*lum[beginIndex]:
                #check for +ve gradient
                upTurn = True
                print('UpTurn has occurred at {0:.2f}'.format(tim[jj]))
                break

        if endBurst == True: break
        
        stopIndex = jj
        
        currentPeakLum = lum[beginIndex]
        currentPersLum = lum[recordStartIndex]
        
        peakTime.append(tim[beginIndex]) #store the time of the peak in structure
        peakIndex.append(beginIndex)
        startIndex.append(recordStartIndex)
        startTimes.append(tim[recordStartIndex])
        peakLum.append(currentPeakLum)
        persLum.append(currentPersLum)

        endIndex.append(stopIndex)
        endTimes.append(tim[stopIndex])
        
        burstRiseTimes = tim[recordStartIndex:beginIndex+2]
        burstRiseLums = lum[recordStartIndex:beginIndex+2]
        
        t10.append( findAtRiseFrac(burstRiseTimes, burstRiseLums,
          currentPeakLum, currentPersLum, 0.10) - peakTime[-1])
        t25.append( findAtRiseFrac(burstRiseTimes, burstRiseLums,
          currentPeakLum, currentPersLum, 0.25) - peakTime[-1])
        t90.append( findAtRiseFrac(burstRiseTimes, burstRiseLums,
          currentPeakLum, currentPersLum, 0.90) - peakTime[-1])


        burstLum = np.array(lum[recordStartIndex:stopIndex])
        burstTim = np.array(tim[recordStartIndex:stopIndex] - tim[beginIndex])
        burstRad = np.array(rad[recordStartIndex:stopIndex])
        
        burstLums.append(burstLum)       
        burstTims.append(burstTim)
        burstRads.append(burstRad)

        upTurns.append(upTurn)
                
        tBurst.append(max(burstTim) - t25[-1])
        
        convexities.append( convexity(burstRiseTimes,
          burstRiseLums, currentPeakLum, currentPersLum) )

        #Tail Fits
        pParams, pData, eParams, eData = burstFits(burstTim, burstLum, 
          0, mdl = None) #PeakTime is 0 at the moment
        
        fitAlpha.append(pParams['al'].value)
        fitDecay.append(eParams['tau'].value)
          
        #Now it is time to find fluence
        fluence = 0.
        for ii in np.arange(recordStartIndex,stopIndex):
          fluence += 0.5*((lum[ii] + lum[ii+1] - 2*currentPersLum)*
            (tim[ii+1] - tim[ii]))
        fluences.append(fluence)
        n = len(peakLum)

        if n > 1:
          tDel.append(peakTime[n-1] - peakTime[n-2])

        if n >= 100:
            print('<separate> CAUTION: more than 100 peaks only analysing first 100')
            print('<separate> Check data file, the data may be a little crazy')
            print('<separate> This is model number %s' % modelID)
            break
    jj+=1
    
    
  for ii in range(0,len(peakIndex)):
    fname = outputDirectory+'bursts/%s/%i.data' % (modelID, ii)
#    print 'writing burst to %s' % fname
    saveArray = zip(burstTims[ii],burstLums[ii],burstRads[ii])
    headString = 'time luminosity radius'
    np.savetxt(fname, saveArray, delimiter=' ',newline='\n',header=headString)
    
  
  
    #now we have arrays of tdels,fluences, and peak times
    #If there are peaks, then return all this data as an output
  
  if peakIndex != []:
    # get the tau array - the ratio of fluence to peak flux
    taus = [fl/p for (fl,p) in zip(fluences,peakLum)]
    #we can now save this in a dictionary
    return {'burstID'   : modelID,         #burst model number
            'num'       : len(peakLum),    #number of bursts
            'pTimes'    : peakTime,        #time of peaks
            'bstart'    : startTimes,      #time of recording start
            'bend'      : endTimes,        #end of burst 
            'peakLum'   : peakLum,         #peak luminosity
            'persLum'   : persLum,         #persistent emission
            'tdel'      : tDel,            #tdel value
            'tau'       : taus,            #tau value
            'fluen'     : fluences,        #fluence value
            'lums'      : burstLums,       #luminosities
            'tims'      : burstTims,       #times
            'rads'      : burstRads,       #rads
            'conv'      : convexities,     #convexities for each burst
            'length'    : tBurst,          #duration of bursts
            't10'       : t10,             #time at 10% burst rise
            't25'       : t25,             #time at 25% burst rise
            't90'       : t90,             #time at 90% burst rise
            'upTurn'    : True in upTurns, #is there an upTurn at burst end
            'fitAlpha'  : fitAlpha,        #Plaw Alpha fir values
            'fitDecay'  : fitDecay,        #exp tail decay timescale
            'endBurst'  : endBurst}        #Burst continues to end of train
    
  #but if there aren't any peaks
  else:
    return {'burstID'  : modelID,   #burst model number
            'num'      : 0.0,       #number of bursts
            'endBurst' : endBurst}  #burst at end



###############################################################################
###############################################################################
#####                                                                     #####
#####                       MODEL ANALYSIS SCRIPT                         #####
#####                                                                     #####
###############################################################################
###############################################################################


def analysis(keyTable, modelDir, outputDir, notAnalysable = [], twinPeaks = [], stableTrans = []):
  '''
  analysis(keyTable, modelDir, outputDir,  notAnalysable = [], twinPeaks = [], stableTrans = []):
  
  
  
  
  #This program will analyse all the rebinned data files to produce a database of
  #tau,tdel,pflux and fluence values
  #It will also produce a mean lightcurve for each source 
  #
  #Jul 2012 - added code to redshift Lacc/Ledd
  '''
  #convert keywords to their variable names
  modelDirectory = modelDir
  outputDirectory = outputDir
  init = keyTable
  
  #Process the model initial conditions table
  modelIDs = [ name.strip() for name in init['name'] ]
  strip = lambda x: x.strip()
  name = np.array(map(strip, init['name']))
  acc = init['acc']
  z = init['z']
  h = init['h']
  lAcc = init['lAcc']
  pul = np.array(map(strip, init['pul']))
  cyc = init['cyc']
  comm = np.array(map(strip, init['comm']))
  
  



  #Now we open and prepare the necessary files
  #Open and read the model info file from Alex

  #dt = [(str('name'),'S5'),(str('acc'),float),(str('z'),float),
  #  (str('h'),float),(str('lAcc'),float)]
  #  #need str() as genfromtxt has limited unicode support

  
  dt = [ (str('name'), 'S4'), (str('acc'), float), (str('z'), float),
         (str('h'), float), (str('lAcc'), float), (str('pul'), 'S20'),
         (str('cyc'), int), (str('comm'), 'S200') ]

  #init = np.genfromtxt(modelDirectory+'MODELS2.txt', dtype = dt)
  init = np.genfromtxt(modelDirectory+'MODELS2.txt', dtype = dt, 
    delimiter=[4,15,8,8,10,20,10,200] )
 




  dbVals = []
  
  #summHead = 'model bursts acc z h lacc/ledd bTime u_bTime pLum u_pLum fluence u_fluence tau u_tau tDel u_tDel conv u_conv rise10 u_rise10 rise25 u_rise25 PLawF0 PLawT0 PLawTs PLawAlpha PLawFp u_PLawF0 u_PLawT0 u_PLawTs u_PLawAlpha u_PLawFp PLawRCSQ exp1F0 u_exp1F0 exp1T0 u_exp1T0 exp1Tau u_exp1Tau exp1Fp u_exp1Fp exp1RCSQ superEddington Flag'
  summVals = []
  
  #---------------------------------------------------------------------------#
  #---------------------------------------------------------------------------#
  #                        BEGIN ANALYSIS LOOP                                #
  #---------------------------------------------------------------------------#
  #---------------------------------------------------------------------------#
  for modelID in modelIDs:
    '''
    '''
    if not os.path.exists(outputDirectory+'bursts/'+str(modelID)):
      os.mkdir(outputDirectory+'bursts/'+str(modelID))
    

    loadFilename = modelDirectory+'xrb'+str(modelID)+'.data'
    burstTime,burstLum,burstRad = np.loadtxt(loadFilename, skiprows=1, 
      unpack=True)
    #remove duplicate times and where lum <= 0

    ok = np.nonzero(np.logical_and((burstTime - np.roll(burstTime,1))!=0,
      burstLum>0))[0]
    burstTime = burstTime[ok]
    burstLum = burstLum[ok]
    burstRad = burstRad[ok]
    #after removal we keep the first time value, rather than averaging
    #as the values only show varation at the 6th sig fig.

    if max(burstLum)>1e39:
      print('%s exceeds Eddington, with peak lum at %e' % 
        (modelID, max(burstLum)) )
      #If the Eddington luminosity is exceeded by convection effects
      #then smooth the train, to get a train more representative of observations
      #use 0.125s as that is commensurate with common RXTE timebins
      #Only downside is sometimes the entire rise can happen in ~0.5s
      burstTime, burstLum, burstRad = smooth(burstTime, burstLum, burstRad,
        0.125)
      superEddington = True
    else:
      superEddington = False
    #--------------#
    # Run Separate #
    #--------------#
    if modelID not in notAnalysable:
      x=separate(burstTime, burstLum, burstRad, modelID, outputDirectory)
    else:
      x = {'num' : 0, 'endBurst' : False, 'burstID' : modelID}
    
    #-----------------# 
    # Set Flag Values #
    #-----------------#
    '''
    Flags are used to indicate where the model may be questionable.
    These are done in base 2. The flags are
    
    0:  No analysis issues
    1:  Burst at end of file (last burst not analysed)
    2:  Shocks occur that cause luminosity to exceed L>10**39 erg s^-1
    4:  Bursts have been cut at a local minimum rather than by luminosity
    8:  Bursts in this train are twin peaked, convexity should not really be 
        considered for these models. Set manually
    16: Rapid bursts with recurrence time less than 100s, 
        This may indicate some bursts are missed, or the observations include 
        multiple bursts Often these missed bursts are low intensity bursts.
    32: Burst not conducive to analysis
    '''
    loc  = np.nonzero(name == modelID)[0]
    lAccErgs = float(1.17e46*acc[loc]) #For 1.4Msun, 10km NS (rest frame)
    flag = int(0)
    if not 'upTurn' in x: x['upTurn'] = False
    if x['endBurst']: flag += 2**0                               #flag 01
    if superEddington: flag += 2**1                              #flag 02
    if x['upTurn'] == True: flag += 2**2                         #flag 04
    if modelID in twinPeaks: flag += 2**3                        #flag 08
    if x['num'] > 1:
      if (np.asarray(x['tdel'][1:])<100).any(): flag += 2**4     #flag 16
    if modelID in notAnalysable: flag += 2**5                    #flag 32
    print('model: %s; flag: %i' % (modelID,flag) )


    #----------------------------------#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # AVG EVALUATION STUFF BEGINS HERE #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #----------------------------------#
    
    if x['num'] != 0: #if there are bursts
      #get average LCs and parameters

      #-----------------------#
      # Save Mean Lightcurves #
      #-----------------------#
      
      #~~~~~~~~THIS NEEDS TO GO BEFORE AVG PARAMETERS~~~~~~~~#
      fullBTime = [a[-1] - a[0] for a in x['tims']]
      longestBurstTime = max(fullBTime)
      allTims = []
      allLums = []
      allRads = []
      for (t,b) in zip(x['tims'],x['bstart']):
        sInd = bn.nanargmax(burstTime >= (b))
        fInd = bn.nanargmax(burstTime >= (b+longestBurstTime))
        if fInd == 0: fInd = len(burstTime)
        allTims.append(burstTime[sInd:fInd]-b+t[0])
        allLums.append(burstLum[sInd:fInd])
        allRads.append(burstRad[sInd:fInd])
   


      #~~~Mean Lightcurve~~~#
      mt, ml, mr, mdl, mdr = avgParams(allTims,allLums,allRads)
      fname = outputDirectory+'bursts/%s/mean.data' % (x['burstID'])
      print('writing burst to %s' % fname)
      saveArray = zip(mt, ml, mdl, mr, mdr)
      headString = 'time luminosity u_luminosity radius u_radius'
      np.savetxt(fname, saveArray, delimiter=' ',newline='\n',header=headString)
      

        
      #----------------------------#
      # Save burst database values #
      #----------------------------#
      #calculate alpha ratio
      alphas = [(lAccErgs+ps)*dt/fl for (ps,dt,fl) in zip(x['persLum'],
        x['tdel'], x['fluen'])]

      for ii in xrange(0, x['num']):
        #add them to the database and sum the lightcurves
        dbVals.append(( x['burstID'],
                        ii,
                        x['bstart'][ii],
                        x['length'][ii],
                        x['fluen'][ii],
                        x['peakLum'][ii],
                        x['persLum'][ii],
                        x['tau'][ii],
                        x['tdel'][ii],
                        x['conv'][ii],
                        x['t10'][ii],
                        x['t25'][ii],
                        x['t90'][ii],
                        x['fitAlpha'][ii],
                        x['fitDecay'][ii],
                        alphas[ii]))
                        
      #Ignore the first burst in train if there are more than 3 bursts
    
      #------------------------#
      # Get average parameters #
      #------------------------#
      if x['num'] >= 3:
        for value in x.itervalues():
          if type(value) == list: value.remove(value[0])
       
      
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
      
      #rise time 10%-90%
      t10t90 = np.array(x['t90']) - np.array(x['t10'])
      r1090 = np.mean(t10t90)
      uR1090 = np.std(t10t90)

      #rise time 25%-90%
      t25t90 = np.array(x['t90']) - np.array(x['t25'])
      r2590 = np.mean(t25t90)
      uR2590 = np.std(t25t90)
      

      if x['num']>=2: 
        tDel = np.mean(x['tdel'][1:])
        uTDel = np.std(x['tdel'][1:])
        alpha = np.mean(alphas[1:])
        uAlpha = np.std(alphas[1:])
      else:
        tDel = 0.
        uTDel = 0.
        alpha = 0.
        uAlpha = 0.
    #No bursts case!!
    else:
      #set all the things to zero!!
      ### OBSERVABLES ###
      burstLength = 0
      uBurstLength = 0

      peakLum = 0
      uPeakLum = 0

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


    
    #---------------------------#
    # Record Summary Properties #
    #---------------------------#
      
    #obtain the location of this pulse in model_info.txt
    loc = np.argmax(name == str(modelID))

    summVals.append((str(modelID),
                     x['num'],
                     acc[loc],
                     z[loc],
                     h[loc],
                     lAcc[loc],
                     pul[loc],
                     cyc[loc],
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
                     flag,
                     ))
  #save data


  summHead=['model','num','acc','z','h','lAcc','pul','cyc','burstLength',
    'uBurstLength','peakLum','uPeakLum','persLum','uPersLum','fluence',
    'uFluence','tau','uTau','tDel','uTDel','conv','uConv','r1090','uR1090',
    'r2590','uR2590','singAlpha','uSingAlpha','singDecay','uSingDecay','alpha',
    'uAlpha','flag']          
  """
  summHead=['model','num','acc','z','h','lAcc','pul','cyc','burstLength',
    'uBurstLength','peakLum','uPeakLum','persLum','uPersLum','fluence',
    'uFluence','tau','uTau','tDel','uTDel','conv','uConv','r1090','uR1090',
    'r2590','uR2590','plawF0','plawT0','plawTs','plawAl','plawFp','plawUF0',
    'plawUT0','plawUTs','plawUAl','plawUFp','plawRedChi','exp1F0','exp1T0',
    'exp1Tau','exp1Fp','exp1UF0', 'exp1UT0','exp1UTau','exp1UFp','exp1RedChi',
    'singAlpha','uSingAlpha','singDecay','uSingDecay','alpha','uAlpha','flag',
    'regime'] 
  """

  dbHead = ['model', 'burst','bstart','btime','fluence','peakLum','persLum',
     'tau','tdel','conv','t10','t25','t90','fitAlpha','fitDecay','alpha']

  ofile = open(outputDirectory+'bursts/db.csv', 'w')
  ofile.write('# ' + ','.join(dbHead) + '\n')
  for line in dbVals:
    ofile.write(','.join(map(repr, line)) + '\n')
  ofile.close()
  
  ofile = open(outputDirectory+'bursts/summ.csv', 'w')
  ofile.write('# ' + ','.join(summHead) + '\n')
  for line in summVals:
    ofile.write(','.join(map(repr, line)) + '\n')
  ofile.close()
  #np.savetxt(outputDirectory+'bursts/db.txt',dbVals, header=dbHead,
  #  fmt=['%s','%i']+11*['%e'])
  #np.savetxt(outputDirectory+'bursts/summ.txt',summVals, header=summHead,
  #  fmt=['%s','%i']+40*['%g']+2*['%i'])
    
  #done
  return summVals, dbVals



