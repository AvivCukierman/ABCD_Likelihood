from numpy import random,log,arange,argsort,exp,array,insert,sqrt,std,load
#from helper_functions import sample_mode
from pickle import dump as save
import numpy
from iminuit import Minuit
from pdb import set_trace as st
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.style.use('atlas')
import matplotlib.mlab as mlab

def ftoa(x):
  return str(round(x,3))

#Asimov:
def asimov(NinA,NinB,NinC,a,b,c,mu,closure=1):
  bkg_A = NinA - a*mu
  bkg_B = NinB - b*mu
  bkg_C = NinC - c*mu
  bkg_D = (bkg_B*bkg_C/bkg_A)/closure #closure = predict/NinD
  #print bkg_A,bkg_B,bkg_C,bkg_D
  NinD = bkg_D+mu
  return NinD

def quadraturesum(x,y):
  return sqrt(pow(x,2)+pow(y,2))

def logL(mu,muS,tauB,tauC,DA,DB,DC,DD,aa,bb,cc):
  #if mu<0 or muS<0 or tauB<0 or tauC<0: return 1000000000000
  if mu+muS<0 or mu*tauB+muS*bb<0 or mu*tauC+muS*cc<0 or mu*tauB*tauC+muS*aa<0: return 1000000000000
  if math.isnan(mu) or math.isnan(muS) or math.isnan(tauB) or math.isnan(tauC): return 1000000000000
  result = -(DD*log(mu+muS)+DB*log(mu*tauB+muS*bb)+DC*log(mu*tauC+muS*cc)+DA*log(mu*tauB*tauC+muS*aa)) \
      +(mu+muS+mu*tauB+muS*bb+mu*tauC+muS*cc+mu*tauB*tauC+muS*aa)
  #print mu,muS,tauB,tauC,DA,DB,DC,DD,aa,bb,cc,result
  return result

def MLE(NinA_given,NinB_given,NinC_given,NinD_given,a,b,c,pseudodata=1,doExact=False):
  mus = []
  lower_errs = []
  upper_errs = []
  if doExact == True: pseudodata = 1000
  for i in range(pseudodata):
    if i==0:
      NinD = NinD_given
      NinB = NinB_given 
      NinC = NinC_given 
      NinA = NinA_given
    else:
      NinD = random.poisson(NinD_given)
      NinB = random.poisson(NinB_given)
      NinC = random.poisson(NinC_given)
      NinA = random.poisson(NinA_given)
    if NinD==0: NinD=0.01
    if NinA==0: NinA=0.01
    if NinB==0: NinB=0.01
    if NinC==0: NinC=0.01
    
    iupper_errs = []
    ilower_errs = []
    imus = []
    #startcond = [NinD/2,NinD/2] #jiggle starting conditions
    startcond = [NinD,0] #jiggle starting conditions
    while True:
      converged = False
      #startcond = random.uniform(0,NinD/2,2) #jiggle starting conditions
      #startcond = [1,0.8]
      #startcond = [1,1.0]
      #startcond = [2,2.0]
      #startcond = [2,2.1]
      #print startcond
      m=Minuit(logL, \
          DA=NinA,fix_DA=True, \
          DB=NinB,fix_DB=True, \
          DC=NinC,fix_DC=True, \
          DD=NinD,fix_DD=True, \
          aa=a,fix_aa=True, \
          bb=b,fix_bb=True, \
          cc=c,fix_cc=True, \
          mu=startcond[0],limit_mu=(0,None),error_mu=NinD*1, \
          muS=startcond[1],limit_muS=(None,None),error_muS=NinD*10, \
          tauB=NinB/NinD,limit_tauB=(0,None),error_tauB=1.0, \
          tauC=NinC/NinD,limit_tauC=(0,None),error_tauC=1.0, \
          errordef=0.5,print_level=0,throw_nan=True) #https://media.readthedocs.org/pdf/iminuit/latest/iminuit.pdf page 10 for logL
      migradres = m.migrad()
      converged = migradres[0]['is_valid']
      if not converged:
        print 'Didn\'t converge, trying again'
        startcond = [NinD/2,NinD/2] #jiggle starting conditions
      else:
        ilower_errs.append(-m.minos(var='muS')['muS']['lower']) #returns a negative
        iupper_errs.append(m.minos(var='muS')['muS']['upper'])
        imus.append(m.values['muS'])
        #print imus[-1],iupper_errs[-1]
        #print('Returning',m.fval)
        #def logL(mu,muS,tauB,tauC,DA,DB,DC,DD,aa,bb,cc):
        #print('Returning',logL(m.values['muS'],NinD,float(NinB)/NinD,float(NinC)/NinD,NinA,NinB,NinC,NinD,a,b,c))
        #print('True',logL(NinD,0,float(NinB)/NinD,float(NinC)/NinD,NinA,NinB,NinC,NinD,a,b,c))
        #converged=False
        #continue
        if len(imus)==1: break
    median = argsort(imus)[0] #median of 1
    #print imus
    #print iupper_errs
    final_mu = imus[median]
    final_lower_err = ilower_errs[median]
    final_upper_err = iupper_errs[median]

    #print('Values',m.values)
    #print('Errors',m.errors)
    #print('Returning',m.fval)
    #print('True',logL(mu_t,muS_t,tauB_t,tauC_t,NinA,NinB,NinC,NinD))
    #m.print_param()

    mus.append(final_mu)
    lower_errs.append(final_lower_err)
    upper_errs.append(final_upper_err)
  if not doExact: return array(mus),array(lower_errs),array(upper_errs),2*array(upper_errs)
  else: 
    #MLE_mu = sample_mode(mus,above=0.5*mus[0])
    mus_sorted = argsort(mus)
    MLE_lower_err = mus[0]-mus[mus_sorted[int(len(mus)*0.16)]] #16th percentile
    MLE_upper_err = mus[mus_sorted[int(len(mus)*0.84)]]-mus[0] #84th percentile
    MLE_2upper_err = mus[mus_sorted[int(len(mus)*0.95)]]-mus[0] #95th percentile
    return mus,MLE_lower_err,MLE_upper_err,MLE_2upper_err

def cl95values(mu_ts,mus,us,ls,sumls,sumus,acut=7,vcut=800):
  if doSysts:
    plt.errorbar(mu_ts,mus,color='black',ls='-',label='MLE Estimate')
    plt.fill_between(mu_ts,mus-ls,mus+us,edgecolor='white',facecolor='#FFFF00',label='$\pm$1-Sigma')
    plt.errorbar(mu_ts,mus+2*us,color='r',ls='-',label='95\% CL (systematics not included)')
    #plt.errorbar(mu_ts,mu_ts,color='black',ls=':',label='Linearity')
    plt.errorbar(mu_ts,mus+sumus,color='black',ls='--',label='$\pm 1 \sigma$ Systematics')
    plt.errorbar(mu_ts,mus-sumls,color='black',ls='--')
    plt.errorbar(mu_ts,mus+2*quadraturesum(us,sumus),color='r',ls='--',label='95\% CL (systematics included)')
    #plt.errorbar(mu_ts,mus-5*ls,color='g',ls='--',label='5-Sigma Discovery!')
    plt.xlabel('True BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
    plt.ylabel('BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
    plt.xlim(0,0.25)
    plt.ylim(0,0.25)
    #plt.ylim(min(mu_ts),)
    plt.legend(loc='upper left')
    plt.savefig(submitDir+'/ABCD_linearity_'+'allSysts'+'_'+str(acut)+'_'+str(vcut)+'.png')
    plt.close()

  result = []
  sumus = quadraturesum(us,sumus)
  sumls = quadraturesum(ls,sumls)
  sigma0 = sumus[0]
  for N,Nsigmax in zip([0,1,2,-1,-2],[1.96,1.73,1.66,2.41,3.05]):
    #while lowerx-(N*sigma0+Nsigmax*lowersigmax)>0:
    lowerCL = mus-N*sigma0-Nsigmax*sumls
    cl95ind = (i for i,x in enumerate(lowerCL) if x > 0.0001).next() 
    x1 = mu_ts[cl95ind-1]
    y1 = lowerCL[cl95ind-1]
    x2 = mu_ts[cl95ind]
    y2 = lowerCL[cl95ind]
    cl95 = x1-y1*(x2-x1)/(y2-y1)
    result.append(cl95)
  print result
  return result

def linearityFit(NinA_data,NinB_data,NinC_data,a,b,c,mu_t_norm,closure=1,aActual=None,bActual=None,cActual=None,mu_t_normActual=None,pseudodata=1):
  #actual numbers based on systematic:
  if not aActual: aActual=a
  if not bActual: bActual=b
  if not cActual: cActual=c
  if not mu_t_normActual: mu_t_normActual=mu_t_norm

  mu_ts = arange(0.000,3.01,0.1)
  mus = []
  us = []
  u2s = []
  ls = []
  for mu_t_hat in mu_ts:
    #mu_t_hat = 1 #true ratio of mu to 0.1
    mu_t = mu_t_hat*mu_t_norm #sig_D = true number of signal in region D
    mu_tActual = mu_t_hat*mu_t_normActual #sig_D = true number of signal in region D
    NinD_data = asimov(NinA_data,NinB_data,NinC_data,aActual,bActual,cActual,mu_tActual,closure=closure)
    mu,l,u,u2 = MLE(NinA_data,NinB_data,NinC_data,NinD_data,a,b,c,doExact=False,pseudodata=pseudodata)
    #print mu_t_hat,mu[0],NinD_data,mu_t/sqrt(mu_t+NinD_data),mu[0]/l
    #if mu_t_hat<0.1 and closure_syst==0: print acut,vcut,2*u/mu_t_norm,closure,closure_err
    mus.append(mu[0]/mu_t_norm) #should be same as mu_t_hat
    us.append(u[0]/mu_t_norm)
    u2s.append(u2[0]/mu_t_norm)
    ls.append(l[0]/mu_t_norm)
  mu_ts = mu_ts*0.1
  mus = array(mus)*0.1
  us = array(us)*0.1
  ls = array(ls)*0.1
  return mu_ts,mus,us,ls

def getMusAndSigmas(NinA_data,NinB_data,NinC_data,vcut=800,acut=7,doSysts=True,reCalc=True,closure_err=0):
  if not reCalc:
    print '<<Trying to load Mus and Sigmas>>'
    outFileName = submitDir+'/musAndSigmas_a'+str(acut)+'_v'+str(vcut)
    if doSysts: outFileName+='_systs'
    outFileName+='.npy'
    try:
      npzfile = numpy.load(outFileName)
      return npzfile['mu_ts'],npzfile['mus'][()],npzfile['us'][()],npzfile['ls'][()]
    except IOError:
      print '<<Couldn\'t load Mus and Sigmas, recalculating>>'
      reCalc=True
  if reCalc:
    mus_systs = {s["name"]:[] for s in systs}
    mu_ts_systs = {s["name"]:[] for s in systs}
    us_systs = {s["name"]:[] for s in systs}
    ls_systs = {s["name"]:[] for s in systs}

    a_systs = {s["name"]:[] for s in systs}
    b_systs = {s["name"]:[] for s in systs}
    c_systs = {s["name"]:[] for s in systs}
    mu_t_norm_systs = {s["name"]:[] for s in systs}

    NinA_MC = VBF_sig_nums["Nominal"][acut,vcut]['sig']['A']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['A']['num']*norm2
    NinB_MC = VBF_sig_nums["Nominal"][acut,vcut]['sig']['B']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['B']['num']*norm2
    NinC_MC = VBF_sig_nums["Nominal"][acut,vcut]['sig']['C']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['C']['num']*norm2
    NinD_MC = VBF_sig_nums["Nominal"][acut,vcut]['sig']['D']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['D']['num']*norm2
    a = NinA_MC/NinD_MC #MC sig_A/sig_D
    b = NinB_MC/NinD_MC #MC sig_B/sig_D
    c = NinC_MC/NinD_MC #MC sig_C/sig_D
    mu_t_norm = NinD_MC*lmult #MC expected signal BR = 0.1
    a_systs["Nominal"] = a
    b_systs["Nominal"] = b
    c_systs["Nominal"] = c
    mu_t_norm_systs["Nominal"] = mu_t_norm

    mu_ts,mus,us,ls = linearityFit(NinA_data,NinB_data,NinC_data,a,b,c,mu_t_norm,closure=1,aActual=a,bActual=b,cActual=c,mu_t_normActual=mu_t_norm,pseudodata=1)
    mu_ts_systs["Nominal"] = mu_ts
    mus_systs["Nominal"] = mus
    us_systs["Nominal"] = us
    ls_systs["Nominal"] = ls

    if doSysts:
      for s in systs:
        systName = s["name"]
        for v in s["vars"]:
          if systName=='Nominal': continue
          if systName=='Closure': thisVar = "Nominal" 
          else: thisVar = systName+'_'+v

          NinA_MC = VBF_sig_nums[thisVar][acut,vcut]['sig']['A']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['A']['num']*norm2
          NinB_MC = VBF_sig_nums[thisVar][acut,vcut]['sig']['B']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['B']['num']*norm2
          NinC_MC = VBF_sig_nums[thisVar][acut,vcut]['sig']['C']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['C']['num']*norm2
          NinD_MC = VBF_sig_nums[thisVar][acut,vcut]['sig']['D']['num']*norm1+ggH_sig_nums[acut,vcut]['sig']['D']['num']*norm2
          a = NinA_MC/NinD_MC #MC sig_A/sig_D
          b = NinB_MC/NinD_MC #MC sig_B/sig_D
          c = NinC_MC/NinD_MC #MC sig_C/sig_D
          #b = float(NinA_data)/NinC_data
          #c = float(NinA_data)/NinB_data
          #a = b*c 
          mu_t_norm = NinD_MC*lmult #MC expected signal BR = 0.1
          a_systs[systName].append(a)
          b_systs[systName].append(b)
          c_systs[systName].append(c)
          mu_t_norm_systs[systName].append(mu_t_norm)

      for s in systs:
        systName = s["name"]
        if systName=='Nominal': continue
        symmetric = (len(s["vars"])==1)
        for iVar in range(len(a_systs[systName])):
          a = a_systs["Nominal"]
          b = b_systs["Nominal"]
          c = c_systs["Nominal"]
          mu_t_norm = mu_t_norm_systs["Nominal"]
          aActual = a_systs[systName][iVar]
          bActual = b_systs[systName][iVar]
          cActual = c_systs[systName][iVar]
          mu_t_normActual = mu_t_norm_systs[systName][iVar]
          #print systName,iVar
          #print a,b,c,mu_t_norm
          #print aActual,bActual,cActual,mu_t_normActual

          if systName=='Closure' and iVar==0: closure=1-closure_err
          elif systName=='Closure' and iVar==1: closure=1+closure_err
          else: closure=1
          mu_ts,mus,us,ls = linearityFit(NinA_data,NinB_data,NinC_data,a,b,c,mu_t_norm,closure=closure,aActual=aActual,bActual=bActual,cActual=cActual,mu_t_normActual=mu_t_normActual,pseudodata=1)
          mus_systs[systName].append(mus)
          mu_ts_systs[systName].append(mu_ts)
          us_systs[systName].append(us)
          ls_systs[systName].append(ls)
          if symmetric:
            mus_systs[systName].append(2*mus_systs["Nominal"]-mus) #assume symmetric

    for s in systs:
      systName = s["name"]
      if not doSysts and not systName=="Nominal": continue
      mu_ts = mu_ts_systs["Nominal"]
      mus = mus_systs["Nominal"]
      us = us_systs["Nominal"]
      ls = ls_systs["Nominal"]

      plt.errorbar(mu_ts,mus,color='black',ls='-',label='MLE Estimate')
      plt.fill_between(mu_ts,mus-ls,mus+us,edgecolor='white',facecolor='#FFFF00',label='$\pm$1-Sigma')
      #plt.errorbar(mu_ts,mu_ts,color='black',ls=':',label='Linearity')
      plt.errorbar(mu_ts,mus+2*us,color='r',ls='--',label='95\% Confidence Limit')
      plt.errorbar(mu_ts,mus-5*ls,color='g',ls='--',label='5-Sigma Discovery!')
      if not systName=='Nominal':
        plt.errorbar(mu_ts,mus_systs[systName][0],color='black',ls='--',label='$\pm 1 \sigma$ Systematic')
        plt.errorbar(mu_ts,mus_systs[systName][1],color='black',ls='--')
      plt.xlabel('True BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
      plt.ylabel('BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
      plt.xlim(0,0.25)
      plt.ylim(0,0.25)
      #plt.ylim(min(mu_ts),)
      plt.legend(loc='upper left')
      if not systName=='Nominal': plt.savefig(submitDir+'/ABCD_linearity_'+systName+'_'+str(acut)+'_'+str(vcut)+'.png')
      else: plt.savefig(submitDir+'/ABCD_linearity_'+str(acut)+'_'+str(vcut)+'.png')
      plt.close()

    outFileName = submitDir+'/musAndSigmas_a'+str(acut)+'_v'+str(vcut)
    if doSysts: outFileName+='_systs'
    outFileName+='.npy'
    with open(outFileName,'wb') as outFile:
      numpy.savez(outFile,mu_ts=mu_ts,mus=mus_systs,us=us_systs,ls=ls_systs)
    return mu_ts,mus_systs,us_systs,ls_systs

def doMLE(doSysts,doMCUncertainty,doLikelihoodDist):
  closures = {(a,v): [None,None] for a in amasscuts for v in VBFmasscuts}
  twosigmas = {(a,v): [] for a in amasscuts for v in VBFmasscuts}
  #fivesigmas = {(a,v): None for a in amasscuts for v in VBFmasscuts}
  for vcut in VBFmasscuts:
    for acut in amasscuts:
      print vcut,acut
      NinA_data = data_nums[acut,vcut]['sig']['num']['A']*lmult #actual data
      NinB_data = data_nums[acut,vcut]['sig']['num']['B']*lmult #actual data
      NinC_data = data_nums[acut,vcut]['sig']['num']['C']*lmult #actual data
      NinD_data = data_nums[acut,vcut]['sig']['num']['D']*lmult #actual data
      closure = data_nums[acut,vcut]['val']['closure'] #validation data
      closure_err = data_nums[acut,vcut]['val']['closure_err'] #validation data
      closures[acut,vcut] = [closure,closure_err]
      if NinA_data==0 or NinB_data==0 or NinC_data==0: continue

      #mu_t = 0 #no signal->expected 95% confidence limit
      #NinD_data = asimov(NinA_data,NinB_data,NinC_data,a,b,c,mu_t,closure=1)
      #cl95 = cl95Values(NinA_data,NinB_data,NinC_data,NinD_data,a,b,c,mu_t_norm)
      #twosigmas[acut,vcut] = [c*0.1 for c in cl95]

      #
      #syts
      #
      mu_ts,mus,ls,us = getMusAndSigmas(NinA_data,NinB_data,NinC_data,vcut=vcut,acut=acut,doSysts=doSysts,reCalc=False,closure_err=closure_err)
      sumls = ls['Nominal']*0
      sumus = us['Nominal']*0
      if doSysts:
        for s in systs:
          systName = s['name']
          if systName=='Nominal': continue
          sumls = quadraturesum(sumls,mus["Nominal"]-mus[systName][0])
          sumus = quadraturesum(sumus,-mus["Nominal"]+mus[systName][1])
      cl95s = cl95values(mu_ts,mus['Nominal'],us['Nominal'],ls['Nominal'],sumls,sumus,acut=acut,vcut=vcut)
      twosigmas[acut,vcut] = cl95s

      if not (doMCUncertainty or doLikelihoodDist): continue
      for mu_t_hat in [0,0.5,1.0]:
        print 'mu_t_hat',mu_t_hat
        #mu_t_hat = 1 #true ratio of mu to 0.1
        mu_t = mu_t_hat*mu_t_norm #sig_D = true number of signal in region D
        NinD_data = asimov(NinA_data,NinB_data,NinC_data,a,b,c,mu_t)
        print 'Generating pseudodata'
        if doMCUncertainty: 
          mus = []
          mu,lower_err,upper_err = MLE(NinA_data,NinB_data,NinC_data,NinD_data,a,b,c,pseudodata=1)
          mus.append(mu[0])
          for _ in range(2000):
            NinA_MC = random.poisson(VBF_sig_nums[acut,vcut]['sig']['A']['num'])*norm1+random.poisson(ggH_sig_nums[acut,vcut]['sig']['A']['num'])*norm2
            NinB_MC = random.poisson(VBF_sig_nums[acut,vcut]['sig']['B']['num'])*norm1+random.poisson(ggH_sig_nums[acut,vcut]['sig']['B']['num'])*norm2
            NinC_MC = random.poisson(VBF_sig_nums[acut,vcut]['sig']['C']['num'])*norm1+random.poisson(ggH_sig_nums[acut,vcut]['sig']['C']['num'])*norm2
            NinD_MC = random.poisson(VBF_sig_nums[acut,vcut]['sig']['D']['num'])*norm1+random.poisson(ggH_sig_nums[acut,vcut]['sig']['D']['num'])*norm2
            a = NinA_MC/NinD_MC #MC sig_A/sig_D
            b = NinB_MC/NinD_MC #MC sig_B/sig_D
            c = NinC_MC/NinD_MC #MC sig_C/sig_D
            mu,lower_err_rand,upper_err_rand = MLE(NinA_data,NinB_data,NinC_data,NinD_data,a,b,c,pseudodata=1)
            mus.append(mu[0])
        if doLikelihoodDist: mus,lower_err,upper_err = MLE(NinA_data,NinB_data,NinC_data,NinD_data,a,b,c,pseudodata=10000)
        mus = array(mus)/mu_t_norm
        mus = mus*0.1
        lower_err = lower_err[0]/mu_t_norm*0.1
        upper_err = upper_err[0]/mu_t_norm*0.1
        binwidth = 0.5*std(mus)
        n,bins = numpy.histogram(mus,normed=True,bins=numpy.arange(0, max(mus)+binwidth, binwidth))
        n = insert(n,0,0)
        n = insert(n,len(n),0)
        bins = insert(bins,len(bins),bins[-1]+binwidth)
        n = n/sum(n)
        plt.plot(bins,n,drawstyle='steps',ls='-',color='b',label='Pseudo $\mu$')
        maxn = max(n[1:])
        plt.plot([mus[0],mus[0]],[0,maxn],ls='--',color='g',label='True $\mu$')
        plt.plot([mus[0]-lower_err,mus[0]+upper_err],[maxn*0.61,maxn*0.61],ls='--',color='g')
        #plt.plot([mus[i],mus[i]],[0,max(Ls)],ls='--',color='r',label='MLE $\mu$')
        plt.xlabel('BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
        plt.ylabel('a.u.')
        xlim = max(2*mus[0],mus[0]+3*std(mus),mus[0]+upper_err*1.1)
        plt.xlim(0,xlim)
        plt.ylim(0,maxn*1.1)
        plt.legend(loc='upper right')
        if doMCUncertainty:
          plt.text(xlim/20,maxn,'Upper Error: '+ftoa(upper_err)+'$\oplus$'+ftoa(std(mus))+'='+ftoa(sqrt(pow(upper_err,2)+pow(std(mus),2))))
          plt.text(xlim/20,0.9*maxn,'Lower Error: '+ftoa(lower_err)+'$\oplus$'+ftoa(std(mus))+'='+ftoa(sqrt(pow(lower_err,2)+pow(std(mus),2))))
          plt.savefig(submitDir+'/ABCD_likelihood_MCUnc_mu'+str(int(mu_t_hat*10))+'_'+str(acut)+'_'+str(vcut)+'.png')
        else:
          plt.savefig(submitDir+'/test_ABCD_likelihood_mu'+str(int(mu_t_hat*10))+'_'+str(acut)+'_'+str(vcut)+'.png')
        plt.close()
  return closures,twosigmas

def calcValRatio(acut,vcut):
  NinA_data_sig = data_nums[acut,vcut]['sig']['num']['A']*lmult #actual data
  NinB_data_sig = data_nums[acut,vcut]['sig']['num']['B']*lmult #actual data
  NinC_data_sig = data_nums[acut,vcut]['sig']['num']['C']*lmult #actual data
  NinD_data_sig = data_nums[acut,vcut]['sig']['num']['D']*lmult #actual data
  NinA_err_sig = data_nums[acut,vcut]['sig']['err']['A']*lmult #actual data
  NinB_err_sig = data_nums[acut,vcut]['sig']['err']['B']*lmult #actual data
  NinC_err_sig = data_nums[acut,vcut]['sig']['err']['C']*lmult #actual data
  NinD_err_sig = data_nums[acut,vcut]['sig']['err']['D']*lmult #actual data
  NinA_data_val = data_nums[acut,vcut]['val']['num']['A']*lmult #actual data
  NinB_data_val = data_nums[acut,vcut]['val']['num']['B']*lmult #actual data
  NinC_data_val = data_nums[acut,vcut]['val']['num']['C']*lmult #actual data
  NinD_data_val = data_nums[acut,vcut]['val']['num']['D']*lmult #actual data
  NinA_err_val = data_nums[acut,vcut]['val']['err']['A']*lmult #actual data
  NinB_err_val = data_nums[acut,vcut]['val']['err']['B']*lmult #actual data
  NinC_err_val = data_nums[acut,vcut]['val']['err']['C']*lmult #actual data
  NinD_err_val = data_nums[acut,vcut]['val']['err']['D']*lmult #actual data
  ratioA = float(NinA_data_sig)/NinA_data_val
  errA = ratioA*sqrt(pow(NinA_err_sig/NinA_data_sig,2)+pow(NinA_err_val/NinA_data_val,2)) 
  ratioB = float(NinB_data_sig)/NinB_data_val
  errB = ratioB*sqrt(pow(NinB_err_sig/NinB_data_sig,2)+pow(NinB_err_val/NinB_data_val,2)) 
  ratioC = float(NinC_data_sig)/NinC_data_val
  errC = ratioC*sqrt(pow(NinC_err_sig/NinC_data_sig,2)+pow(NinC_err_val/NinC_data_val,2)) 
  ratioD = float(NinD_data_sig)/NinD_data_val
  errD = ratioD*sqrt(pow(NinD_err_sig/NinD_data_sig,2)+pow(NinD_err_val/NinD_data_val,2)) 
  return {'A':(ratioA,errA),'B':(ratioB,errB),'C':(ratioC,errC),'D':(ratioD,errD)}
  #closure = data_nums[acut,vcut]['val']['closure'] #validation data
  #closure_err = data_nums[acut,vcut]['val']['closure_err'] #validation data

def checkValRatio(amasscuts,VBFmasscuts):
  regions = ['A','B','C','D']
  colors = ['b','r','g','black']
  ratios = {(a,v):{k:(0,0) for k in regions} for a in amasscuts for v in VBFmasscuts}
  for a in amasscuts:
    for v in VBFmasscuts:
      ratios[a,v] = calcValRatio(a,v)
      print a,v,ratios[a,v]

  for a in amasscuts:
    for k,c in zip(regions,colors):
      y = [ratios[a,v][k][0] for v in VBFmasscuts]
      yerr = [ratios[a,v][k][1] for v in VBFmasscuts]
      plt.errorbar(VBFmasscuts,y,yerr=yerr,color=c,label=k)
    plt.ylim(0,None)
    plt.legend(loc='upper right')
    plt.xlabel('VBF mass cut (GeV)')
    plt.ylabel('Signal/validation ratio')
    plt.savefig(submitDir+'/'+'valratio_a'+str(a)+'.png')
    plt.close()

  for v in VBFmasscuts:
    for k,c in zip(regions,colors):
      y = [ratios[a,v][k][0] for a in amasscuts]
      yerr = [ratios[a,v][k][1] for a in amasscuts]
      plt.errorbar(amasscuts,y,yerr=yerr,color=c,label=k)
    plt.ylim(0,None)
    plt.legend(loc='upper right')
    plt.xlabel('$a$ mass cut (GeV)')
    plt.ylabel('Signal/validation ratio')
    plt.savefig(submitDir+'/'+'valratio_v'+str(v)+'.png')
    plt.close()


submitDir = 'AL_output'
#L2016 = 32.8616 ifb
lmult = 1 #luminosity_multiplier
amasscuts = range(7,9,1)
VBFmasscuts = range(200,1001,100)
#amasscuts = range(3,12,1)
#VBFmasscuts = [800] 
data_nums = load('ABCD_nums_VBF2_data16_periodAL_3.9.17.p')
#checkValRatio(amasscuts,VBFmasscuts)
VBF_sig_nums = load('ABCD_nums_VBF2_VBF_a30_ggyy_3.10.17.p')
ggH_sig_nums = load('ABCD_nums_VBF2_ggH_a30_ggyy_3.10.17.p')

systs = json.load(open('jet_systs.json','rb'))

MC_params = json.load(open('MC_ABCD.json','rb'))
MC1 = MC_params['VBF_a30_ggyy']
MC2 = MC_params['ggH_a30_ggyy']
norm1 = MC1['norm']
norm2 = MC2['norm']

doSysts = True
doOpt = False
doMCUncertainty = False
doLikelihoodDist = False 

recalc = True
rewrite = True
if not recalc:
  closures = load(submitDir+'/closures_ABCD.p')
  twosigmas = load(submitDir+'/twosigmas_brazil_ABCD.p')
  if not doSysts: twosigmas = load(submitDir+'/twosigmas_brazil_ABCD.p')
  else: twosigmas = load(submitDir+'/twosigmas_brazil_systs_ABCD.p')
else:
  closures,twosigmas = doMLE(doSysts,doMCUncertainty,doLikelihoodDist)
  if rewrite:
    if not doSysts: save(twosigmas,open(submitDir+'/twosigmas_brazil_ABCD.p','wb'))
    else: save(twosigmas,open(submitDir+'/twosigmas_brazil_systs_ABCD.p','wb'))
    #save(closures,open(submitDir+'/closures_ABCD.p','wb'))
    #save(twosigmas,open(submitDir+'/twosigmas_ABCD.p','wb'))
    #save(fivesigmas,open(submitDir+'/fivesigmas_ABCD.p','wb'))

doBrazil = True
if doBrazil:
  for v in VBFmasscuts:
    plt.errorbar(amasscuts,[twosigmas[a,v][0] for a in amasscuts],label='VBF $M_{jj}$ cut = '+str(v)+' GeV',color='black',ls='--')
    plt.fill_between(amasscuts,[twosigmas[a,v][4] for a in amasscuts],[twosigmas[a,v][2] for a in amasscuts],edgecolor='#FFFF00',facecolor='#FFFF00',label='$\pm$2-Sigma')
    plt.fill_between(amasscuts,[twosigmas[a,v][3] for a in amasscuts],[twosigmas[a,v][1] for a in amasscuts],edgecolor='#00FF00',facecolor='#00FF00',label='$\pm$1-Sigma')
    plt.xlim(0,max(amasscuts)+min(amasscuts))
    plt.ylim(0,0.12)
    plt.legend(loc='upper right')
    plt.xlabel('$|M_{jj}-M_{\gamma\gamma}|$ Cut (GeV)')
    plt.ylabel('95\% Confidence Limit on BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
    if not doSysts: plt.savefig(submitDir+'/ABCD_twosigma_brazil_v'+str(v)+'.png')
    else: plt.savefig(submitDir+'/ABCD_twosigma_brazil_systs_v'+str(v)+'.png')
    plt.close()

  for a in amasscuts:
    plt.errorbar(VBFmasscuts,[twosigmas[a,v][0] for v in VBFmasscuts],label='$|M_{jj}-M_{\gamma\gamma}|$ cut = '+str(a)+' GeV',color='black',ls='--')
    plt.fill_between(VBFmasscuts,[twosigmas[a,v][4] for v in VBFmasscuts],[twosigmas[a,v][2] for v in VBFmasscuts],edgecolor='#FFFF00',facecolor='#FFFF00',label='$\pm$2-Sigma')
    plt.fill_between(VBFmasscuts,[twosigmas[a,v][3] for v in VBFmasscuts],[twosigmas[a,v][1] for v in VBFmasscuts],edgecolor='#00FF00',facecolor='#00FF00',label='$\pm$1-Sigma')
    plt.xlim(0,max(VBFmasscuts)+min(VBFmasscuts))
    plt.ylim(0,0.12)
    plt.legend(loc='upper right')
    plt.xlabel('VBF $M_{jj}$ Cut (GeV)')
    plt.ylabel('95\% Confidence Limit on BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
    if not doSysts: plt.savefig(submitDir+'/ABCD_twosigma_brazil_a'+str(a)+'.png')
    else: plt.savefig(submitDir+'/ABCD_twosigma_brazil_systs_a'+str(a)+'.png')
    plt.close()

if doOpt:
    for v in VBFmasscuts:
      plt.errorbar(amasscuts,[closures[a,v][0] for a in amasscuts],yerr=[closures[a,v][1] for a in amasscuts],label='VBF $M_{jj}$ cut = '+str(v)+' GeV')
      plt.plot([0,100],[1,1],color='black',ls='--')
      plt.xlim(0,max(amasscuts)+min(amasscuts))
      plt.legend(loc='upper right')
      plt.xlabel('$|M_{jj}-M_{\gamma\gamma}|$ Cut (GeV)')
      plt.ylabel('Closure in Validation Region')
      plt.savefig(submitDir+'/ABCD_closure_v'+str(v)+'.png')
      plt.close()

    for v in VBFmasscuts:
      plt.errorbar(amasscuts,[twosigmas[a,v] for a in amasscuts],label='VBF $M_{jj}$ cut = '+str(v)+' GeV')
      plt.xlim(0,max(amasscuts)+min(amasscuts))
      plt.legend(loc='upper right')
      plt.xlabel('$|M_{jj}-M_{\gamma\gamma}|$ Cut (GeV)')
      plt.ylabel('95\% Confidence Limit on BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
      plt.savefig(submitDir+'/ABCD_twosigma_v'+str(v)+'.png')
      plt.close()

    for v in VBFmasscuts:
      plt.errorbar(amasscuts,[fivesigmas[a,v] for a in amasscuts],label='VBF $M_{jj}$ cut = '+str(v)+' GeV')
      plt.xlim(0,max(amasscuts)+min(amasscuts))
      plt.legend(loc='upper right')
      plt.xlabel('$|M_{jj}-M_{\gamma\gamma}|$ Cut (GeV)')
      plt.ylabel('Minimum BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$ for 5-$\sigma$ Discovery')
      plt.savefig(submitDir+'/ABCD_fivesigma_v'+str(v)+'.png')
      plt.close()

    for a in amasscuts:
      plt.errorbar(VBFmasscuts,[closures[a,v][0] for v in VBFmasscuts],yerr=[closures[a,v][1] for v in VBFmasscuts],label='$|M_{jj}-M_{\gamma\gamma}|$ cut = '+str(a)+' GeV')
      plt.plot([0,1000],[1,1],color='black',ls='--')
      plt.xlim(0,max(VBFmasscuts)+min(VBFmasscuts))
      plt.legend(loc='upper right')
      plt.xlabel('$|M_{jj}-M_{\gamma\gamma}|$ Cut')
      plt.xlabel('VBF $M_{jj}$ Cut (GeV)')
      plt.ylabel('Closure in Validation Region')
      plt.savefig(submitDir+'/ABCD_closure_a'+str(a)+'.png')
      plt.close()

    for a in amasscuts:
      plt.errorbar(VBFmasscuts,[twosigmas[a,v] for v in VBFmasscuts],label='$|M_{jj}-M_{\gamma\gamma}|$ cut = '+str(a)+' GeV')
      plt.plot([0,100],[1,1],color='black',ls='--')
      plt.xlim(0,max(VBFmasscuts)+min(VBFmasscuts))
      plt.ylim(0,0.1)
      plt.legend(loc='upper right')
      plt.xlabel('VBF $M_{jj}$ Cut (GeV)')
      plt.ylabel('95\% Confidence Limit on BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$')
      plt.savefig(submitDir+'/ABCD_twosigma_a'+str(a)+'.png')
      plt.close()

    for a in amasscuts:
      plt.errorbar(VBFmasscuts,[fivesigmas[a,v] for v in VBFmasscuts],label='$|M_{jj}-M_{\gamma\gamma}|$ cut = '+str(a)+' GeV')
      plt.plot([0,100],[1,1],color='black',ls='--')
      plt.xlim(0,max(VBFmasscuts)+min(VBFmasscuts))
      plt.ylim(0,0.2)
      plt.legend(loc='upper right')
      plt.xlabel('$|M_{jj}-M_{\gamma\gamma}|$ Cut')
      plt.xlabel('VBF $M_{jj}$ Cut (GeV)')
      plt.ylabel('Minimum BR$(H\\rightarrow aa \\rightarrow gg\gamma\gamma)$ for 5-$\sigma$ Discovery')
      plt.savefig(submitDir+'/ABCD_fivesigma_a'+str(a)+'.png')
      plt.close()

