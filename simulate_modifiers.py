import numpy as np
import numpy
from numpy.random import poisson
from math import log,exp
import sys
from numpy.random import binomial
from numpy.random import choice
from numpy.random import exponential
from scipy.special import gamma
import matplotlib.pyplot as plt
import pandas as pd

##


def short_tailed_exponential(x,s,b):
    return (1/s)*(1/(gamma(1+1/b)))*np.exp(-(x/s)**b)


def random_short_tailed_exponential(short_tailed_cdf,discretization,size):
    samples=np.random.uniform(0,1,size)
    indices=np.searchsorted(short_tailed_cdf,samples)
    vals=discretization[indices]
    #vals=np.asarray([discretization[np.abs(samples[i]-short_tailed_cdf).argmin()] for i in range(0,len(samples))])
    return vals


def evolve_modifier_exp_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,mu,tmax,teq):
    lineages=np.asarray([[0,N]]) #first column  fitness, second column population size
    modifier_lineages=np.asarray([[0,0]])
    X_final=0
    Um=0
    tfix=0
    for t in range(0,tmax+teq):
        if t==teq:
            X_final=0
            Um=mu
            
    
        #deterministic growth
        population_size=(np.sum(np.multiply(np.exp(lineages[:,0]),lineages[:,1]))+np.sum(np.multiply(np.exp(modifier_lineages[:,0]),modifier_lineages[:,1])))
        expected_sizes=np.multiply(np.exp(lineages[:,0]),lineages[:,1])*(N/population_size)
        expected_modifier_sizes=np.multiply(np.exp(modifier_lineages[:,0]),modifier_lineages[:,1])*(N/population_size)
        
        #poisson sampling non-mutants
        sizes=poisson((1-U1-U2-Um)*expected_sizes)
        modifier_sizes=poisson((1-U1-U2-U1_m-U2_m)*expected_modifier_sizes)

        #how many mutants
        number_neutral_modifier_mutants=binomial(np.sum(expected_sizes),Um)
        number_beneficial_mutants=binomial(np.sum(expected_sizes),U1)
        number_deleterious_mutants=binomial(np.sum(expected_sizes),U2)
        number_beneficial_modifier_mutants=binomial(np.sum(expected_modifier_sizes),U1)
        number_deleterious_modifier_mutants=binomial(np.sum(expected_modifier_sizes),U2)
        number_beneficial_modifier_mutants_perturb=binomial(np.sum(expected_modifier_sizes),U1_m)
        
        #who mutates
        new_beneficial_lineages=np.add(choice(lineages[:,0],size=number_beneficial_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0))),exponential(s1,size=number_beneficial_mutants))
        #new_deleterious_lineages=np.add(choice(lineages[:,0],size=number_deleterious_mutants,replace=True,p=expected_sizes/np.sum(expected_sizes)),-1*exponential(s2,size=number_deleterious_mutants))
        new_deleterious_lineages=np.add(choice(lineages[:,0],size=number_deleterious_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0))),s2*np.ones(number_deleterious_mutants))       
        new_neutral_modifier_lineages=choice(lineages[:,0],size=number_neutral_modifier_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0)))
        new_beneficial_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_beneficial_modifier_mutants,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),exponential(s1,size=number_beneficial_modifier_mutants))
        new_beneficial_modifier_lineages_perturb=np.add(choice(modifier_lineages[:,0],size=number_beneficial_modifier_mutants_perturb,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),s1_m*np.ones(number_beneficial_modifier_mutants_perturb))
        #new_deleterious_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_deleterious_modifier_mutants,replace=True,p=expected_modifier_sizes/np.sum(expected_modifier_sizes)),-1*exponential(s2_m,size=number_deleterious_modifier_mutants))
        new_deleterious_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_deleterious_modifier_mutants,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),s2*np.ones(number_deleterious_modifier_mutants))

        new_lineages=np.hstack((np.dstack(((new_beneficial_lineages),np.ones(number_beneficial_mutants))),np.dstack(((new_deleterious_lineages),np.ones(number_deleterious_mutants)))))
        new_modifier_lineages=np.hstack((np.dstack(((new_beneficial_modifier_lineages),np.ones(number_beneficial_modifier_mutants))),np.dstack(((new_beneficial_modifier_lineages_perturb),np.ones(number_beneficial_modifier_mutants_perturb))),np.dstack(((new_deleterious_modifier_lineages),np.ones(number_deleterious_modifier_mutants))),np.dstack(((new_neutral_modifier_lineages),np.ones(number_neutral_modifier_mutants)))))

        #update vectors
        lineages[:,1]=sizes
        lineages=np.delete(lineages,np.where(sizes==0)[0],axis=0)
        modifier_lineages[:,1]=modifier_sizes
        #if np.shape(modifier_lineages)[0]>1:
        modifier_lineages=np.delete(modifier_lineages,np.where(modifier_sizes==0)[0],axis=0) #this will delete array
        lineages=np.vstack((new_lineages[0],lineages))
        modifier_lineages=np.vstack((new_modifier_lineages[0],modifier_lineages))

        #remove minimum fitness
        X_bar=(np.multiply(lineages[:,0],lineages[:,1]).sum()+np.multiply(modifier_lineages[:,0],modifier_lineages[:,1]).sum())/(np.sum(lineages[:,1])+np.sum(modifier_lineages[:,1]))
        X_final+=X_bar
        lineages[:,0]=lineages[:,0]-X_bar
        modifier_lineages[:,0]=modifier_lineages[:,0]-X_bar

        if len(modifier_lineages)==0:
            modifier_lineages=np.asarray([[0,0]])


        #check for when modifier fixes
        if np.sum(lineages[:,1])<1:
            tfix= t-teq
            break
    v=(X_final)/(tmax-teq)

    return tfix,v

def evolve_modifier_short_tailed_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,mu,tmax,teq):
    lineages=np.asarray([[0,N]]) #first column  fitness, second column population size
    modifier_lineages=np.asarray([[0,0]])
    X_final=0
    Um=0
    tfix=0
    discretization=np.linspace(0,.25,50000)
    dx=discretization[1]-discretization[0]
    b=10
    short_tailed_cdf=np.cumsum(np.asarray(short_tailed_exponential(discretization,s1,b)))*dx
    for t in range(0,tmax+teq):
        if t==teq:
            X_final=0
            Um=mu
    
        #deterministic growth
        population_size=(np.sum(np.multiply(np.exp(lineages[:,0]),lineages[:,1]))+np.sum(np.multiply(np.exp(modifier_lineages[:,0]),modifier_lineages[:,1])))
        expected_sizes=np.multiply(np.exp(lineages[:,0]),lineages[:,1])*(N/population_size)
        expected_modifier_sizes=np.multiply(np.exp(modifier_lineages[:,0]),modifier_lineages[:,1])*(N/population_size)
        
        #poisson sampling non-mutants
        sizes=poisson((1-U1-U2-Um)*expected_sizes)
        modifier_sizes=poisson((1-U1_m-U2_m)*expected_modifier_sizes)

        #how many mutants
        number_neutral_modifier_mutants=binomial(np.sum(expected_sizes),Um)
        number_beneficial_mutants=binomial(np.sum(expected_sizes),U1)
        number_deleterious_mutants=binomial(np.sum(expected_sizes),U2)
        number_beneficial_modifier_mutants=binomial(np.sum(expected_modifier_sizes),U1)
        number_deleterious_modifier_mutants=binomial(np.sum(expected_modifier_sizes),U2_m)
        number_beneficial_modifier_mutants_perturb=binomial(np.sum(expected_modifier_sizes),U1_m)
        
        #who mutates
        new_beneficial_lineages=np.add(choice(lineages[:,0],size=number_beneficial_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0))),random_short_tailed_exponential(short_tailed_cdf,discretization,number_beneficial_mutants))
        #new_deleterious_lineages=np.add(choice(lineages[:,0],size=number_deleterious_mutants,replace=True,p=expected_sizes/np.sum(expected_sizes)),-1*exponential(s2,size=number_deleterious_mutants))
        new_deleterious_lineages=np.add(choice(lineages[:,0],size=number_deleterious_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0))),s2*np.ones(number_deleterious_mutants))       
        new_neutral_modifier_lineages=choice(lineages[:,0],size=number_neutral_modifier_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0)))
        new_beneficial_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_beneficial_modifier_mutants,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),random_short_tailed_exponential(short_tailed_cdf,discretization,number_beneficial_modifier_mutants))
        new_beneficial_modifier_lineages_perturb=np.add(choice(modifier_lineages[:,0],size=number_beneficial_modifier_mutants_perturb,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),s1_m*np.ones(number_beneficial_modifier_mutants_perturb))

        #new_deleterious_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_deleterious_modifier_mutants,replace=True,p=expected_modifier_sizes/np.sum(expected_modifier_sizes)),-1*exponential(s2_m,size=number_deleterious_modifier_mutants))
        new_deleterious_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_deleterious_modifier_mutants,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),s2_m*np.ones(number_deleterious_modifier_mutants))

        new_lineages=np.hstack((np.dstack(((new_beneficial_lineages),np.ones(number_beneficial_mutants))),np.dstack(((new_deleterious_lineages),np.ones(number_deleterious_mutants)))))
        new_modifier_lineages=np.hstack((np.dstack(((new_beneficial_modifier_lineages),np.ones(number_beneficial_modifier_mutants))),np.dstack(((new_beneficial_modifier_lineages_perturb),np.ones(number_beneficial_modifier_mutants_perturb))),np.dstack(((new_deleterious_modifier_lineages),np.ones(number_deleterious_modifier_mutants))),np.dstack(((new_neutral_modifier_lineages),np.ones(number_neutral_modifier_mutants)))))

        #update vectors
        lineages[:,1]=sizes
        lineages=np.delete(lineages,np.where(sizes==0)[0],axis=0)
        modifier_lineages[:,1]=modifier_sizes
        #if np.shape(modifier_lineages)[0]>1:
        modifier_lineages=np.delete(modifier_lineages,np.where(modifier_sizes==0)[0],axis=0) #this will delete array
        lineages=np.vstack((new_lineages[0],lineages))
        modifier_lineages=np.vstack((new_modifier_lineages[0],modifier_lineages))

        #remove minimum fitness
        X_bar=(np.multiply(lineages[:,0],lineages[:,1]).sum()+np.multiply(modifier_lineages[:,0],modifier_lineages[:,1]).sum())/(np.sum(lineages[:,1])+np.sum(modifier_lineages[:,1]))
        X_final+=X_bar
        lineages[:,0]=lineages[:,0]-X_bar
        modifier_lineages[:,0]=modifier_lineages[:,0]-X_bar

        if len(modifier_lineages)==0:
            modifier_lineages=np.asarray([[0,0]])


        #check for when modifier fixes
        if np.sum(lineages[:,1])<1:
            tfix= t-teq
            break
        
    v=(X_final)/(tmax-teq)

    return tfix,v
def evolve_modifier_delta(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,mu,tmax,teq):
    lineages=np.asarray([[0,N]]) #first column  fitness, second column population size
    modifier_lineages=np.asarray([[0,0]])
    X_final=0
    Um=0
    tfix=0
    for t in range(0,tmax+teq):
        if t==teq:
            X_final=0
            Um=mu
            
    
        #deterministic growth
        population_size=(np.sum(np.multiply(np.exp(lineages[:,0]),lineages[:,1]))+np.sum(np.multiply(np.exp(modifier_lineages[:,0]),modifier_lineages[:,1])))
        expected_sizes=np.multiply(np.exp(lineages[:,0]),lineages[:,1])*(N/population_size)
        expected_modifier_sizes=np.multiply(np.exp(modifier_lineages[:,0]),modifier_lineages[:,1])*(N/population_size)
        
        #poisson sampling non-mutants
        sizes=poisson((1-U1-U2-Um)*expected_sizes)
        modifier_sizes=poisson((1-U1_m-U2_m)*expected_modifier_sizes)

        #how many mutants
        number_neutral_modifier_mutants=binomial(np.sum(expected_sizes),Um)
        number_beneficial_mutants=binomial(np.sum(expected_sizes),U1)
        number_deleterious_mutants=binomial(np.sum(expected_sizes),U2)
        number_beneficial_modifier_mutants=binomial(np.sum(expected_modifier_sizes),U1_m)
        number_deleterious_modifier_mutants=binomial(np.sum(expected_modifier_sizes),U2_m)
        
        #who mutates
        new_beneficial_lineages=np.add(choice(lineages[:,0],size=number_beneficial_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0))),s1*np.ones(number_beneficial_mutants))
        new_deleterious_lineages=np.add(choice(lineages[:,0],size=number_deleterious_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0))),s2*np.ones(number_deleterious_mutants))       
        new_neutral_modifier_lineages=choice(lineages[:,0],size=number_neutral_modifier_mutants,replace=True,p=(expected_sizes+(np.sum(expected_sizes)==0)*np.ones(1))/(np.sum(expected_sizes)+(np.sum(expected_sizes)==0)))
        new_beneficial_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_beneficial_modifier_mutants,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),s1_m*np.ones(number_beneficial_modifier_mutants))
        new_deleterious_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_deleterious_modifier_mutants,replace=True,p=(expected_modifier_sizes+(np.sum(expected_modifier_sizes)==0)*np.ones(1))/(np.sum(expected_modifier_sizes)+(np.sum(expected_modifier_sizes)==0))),s2_m*np.ones(number_deleterious_modifier_mutants))

        new_lineages=np.hstack((np.dstack(((new_beneficial_lineages),np.ones(number_beneficial_mutants))),np.dstack(((new_deleterious_lineages),np.ones(number_deleterious_mutants)))))
        new_modifier_lineages=np.hstack((np.dstack(((new_beneficial_modifier_lineages),np.ones(number_beneficial_modifier_mutants))),np.dstack(((new_deleterious_modifier_lineages),np.ones(number_deleterious_modifier_mutants))),np.dstack(((new_neutral_modifier_lineages),np.ones(number_neutral_modifier_mutants)))))

        #update vectors
        lineages[:,1]=sizes
        lineages=np.delete(lineages,np.where(sizes==0)[0],axis=0)
        modifier_lineages[:,1]=modifier_sizes
        modifier_lineages=np.delete(modifier_lineages,np.where(modifier_sizes==0)[0],axis=0) #this will delete array
        lineages=np.vstack((new_lineages[0],lineages))
        modifier_lineages=np.vstack((new_modifier_lineages[0],modifier_lineages))

        
        if len(modifier_lineages)==0:
            modifier_lineages=np.asarray([[0,0]])

        #remove minimum fitness
        X_bar=(np.multiply(lineages[:,0],lineages[:,1]).sum()+np.multiply(modifier_lineages[:,0],modifier_lineages[:,1]).sum())/(np.sum(lineages[:,1])+np.sum(modifier_lineages[:,1]))
        X_final+=X_bar
        lineages[:,0]=lineages[:,0]-X_bar
        modifier_lineages[:,0]=modifier_lineages[:,0]-X_bar

        #check for when modifier fixes
        if np.sum(lineages[:,1])<1:
            tfix= t-teq
            break
    v=X_final/(tmax-teq)
    return tfix,v


        
def compute_Npfix_delta(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m):
    estimates=[]
    R_target=1/(25*(1/s1)*np.log(s1/U1))
    mu=.0001
    for i in range(0,50):
        tfix,v=evolve_modifier_delta(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,mu,1000000,20000)
        if i>1:
            estimates.append(mu*tfix)
        if i>1:
            print(len(estimates)/np.asarray(estimates).sum())
        if tfix<1:
            tfix=1000000
        mu=min([1e-02,mu*(R_target*tfix)])
        
    Npfix=len(estimates)/np.asarray(estimates).sum()
    return Npfix

def compute_Npfix_exp_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m):
    estimates=[]
    R_target=1/(25*(1/s1)*np.log(s1/U1))
    mu=.0001
    for i in range(0,50):
        tfix,v=evolve_modifier_exp_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,mu,1000000,20000)
        if tfix>1:
            estimates.append(mu*tfix)
        if i>1:
            print(len(estimates)/np.asarray(estimates).sum())
        if tfix<1:
            tfix=1000000
        mu=min([1e-02,mu*(R_target*tfix)])
    Npfix=len(estimates)/np.asarray(estimates).sum()
    return Npfix

def compute_Npfix_short_tailed_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m):
    estimates=[]
    R_target=1/(25*(1/s1)*np.log(s1/U1))
    mu=.0001
    for i in range(0,60):
        tfix,v=evolve_modifier_short_tailed_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,mu,1000000,20000)
        if tfix>1:
            estimates.append(mu*tfix)
        if i>1:
            print(len(estimates)/np.asarray(estimates).sum())
        if tfix<1:
            tfix=1000000
        mu=min([1e-02,mu*(R_target*tfix)])
    Npfix=len(estimates)/np.asarray(estimates).sum(); return Npfix
	
def compute_v_delta(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m):
    mu=0
    outs=[]
    for i in range(0,10):
        tfix,v=evolve_modifier_delta(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,mu,500000,0)
        outs.append(v)
    v_mean=np.asarray(outs).sum()/len(outs)
    return v_mean

def compute_v_exp(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m):
    mu=0
    outs=[]
    for i in range(0,10):
        tfix,v=evolve_modifier_exp_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,0,1000000,20000)
        outs.append(v)
    v_mean=np.asarray(outs).sum()/len(outs)
    return v_mean

def compute_v_short_tailed(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m):
    mu=0
    outs=[]
    for i in range(0,10):
        tfix,v=evolve_modifier_short_tailed_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,0,1000000,20000)
        outs.append(v)
    v_mean=np.asarray(outs).sum()/len(outs)
    return v_mean


if __name__=='__main__':
    N = float(sys.argv[1])
    s1 = float(sys.argv[2])
    U1 = float(sys.argv[3])
    s2 = float(sys.argv[4])
    U2 = float(sys.argv[5])
    mu = float(sys.argv[6])
    s1_m = float(sys.argv[7])
    s2_m = float(sys.argv[8])
    U2_m = float(sys.argv[9])
    job_index = float(sys.argv[10])
    outs=[]; Ubs=np.logspace(-3,-1,3)
    for U1_m in Ubs:
        Npfix=compute_Npfix_short_tailed_perturb(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m)
        outs.append([N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m,Npfix])
    v_b=compute_v_short_tailed(N,s1,U1,s2,U2,s1_m,U1_m,s2_m,U2_m)
    outs.append([v_b])
    e=pd.DataFrame(outs)
    e.to_csv('data/lbs_short_tailed_perturb'+str(job_index)+'.csv')
        



