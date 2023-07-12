import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.special import erfc
from scipy.special import gamma
from scipy.stats import norm
import pandas as pd
from scipy import integrate
from scipy.special import factorial
from theory import *

def run():
    #import simulation data
    S085=pd.read_csv('data/Perturbative Results Exp/lbs_perturb_exp_Beneficial_Modifier_85208618.0.csv')
    S03=pd.read_csv('data/Perturbative Results Exp/lbs_perturb_exp_Beneficial_Modifier_85203.0.csv')
    

    #simulation parameters
    Ub=.00001
    N=10**8
    sb=.01
    Ubms1=np.logspace(-9,-6,100)
    Ubms033=np.logspace(-6,-3.7,100)
    Ubms1s=np.logspace(-9,-6,100)
    Ubms033s=np.logspace(-6,-3.7,100)

    #effective parameters
    v1=.000574
    
    # Solve for xc in the background distribution
    xc1=fsolve_eq_xc_general(.01,.00001,.000574,.1,1,1)
    # Solve for 
    s1=xc1-v1/sb
    Ubeff1=Ub*short_tailed_exponential(s1,sb,1)*(2*np.pi*v1)**0.5

    # Do same thing with new function
    new_xc, new_seff, new_Ueff = calculate_xc_seff_Ueff_beta(v1,sb,Ub,1)
    
    print("Comparing new vs old")
    print("Old:", xc1, s1, Ubeff1)
    print("New:", new_xc, new_seff, new_Ueff)

    #theoretical predictions
    old_Npfixs1=[]
    Npfixs1=[]
    old_Npfixs033=[]
    Npfixs033=[]
    Npfixs033s=[]
    Npfixs1s=[]
    for Ubms, outputs, new_outputs, sm in zip([Ubms1,Ubms033],[old_Npfixs1,old_Npfixs033],[Npfixs1,Npfixs033],[.08618,.03]):
        print("sm=", sm)
        for Ubm in Ubms:
            print("Ubm=", Ubm)
            
            xcm1=fsolve_eq_xc_exact_p(N,s1,Ubeff1,sm,Ubm,v1,xc1,1)
        
            dxc=xcm1-xc1
            Npfix=(xcm1/xc1)*np.exp(-xc1*dxc/v1-dxc**2/(2*v1))
            outputs.append(Npfix)
        
            Npfix = calculate_Npfix_beta_plus_delta(N,sb,Ub,1,sm,Ubm,v1)
            
            
            new_outputs.append(Npfix)
    for Ubm in Ubms1s:
        Npfixs1s.append((Ubm*.08618+Ub*sb)/(Ub*sb))
    for Ubm in Ubms033s:
        Npfixs033s.append((Ubm*.03+Ub*sb)/(Ub*sb))
    
    Npfixs1=np.asarray(Npfixs1); Npfixs033=np.asarray(Npfixs033);
    old_Npfixs1 = np.array(old_Npfixs1)
    old_Npfixs033 = np.array(old_Npfixs033)
    
    #plot simulation and theory
    plt.figure(figsize=(14,12)); plt.xlabel(r"$U_{b}^{1}/U_{0}$",fontsize=65); plt.ylabel(r"$ \tilde{p}_{fix}(\mu \to \mu +\delta \mu)$",fontsize=65); plt.yscale('log'); 
    plt.plot(Ubms1/Ub,Npfixs1,color='cornflowerblue',linewidth=10)
    plt.plot(Ubms033/Ub,Npfixs033,color='blue',linewidth=10)
    plt.plot(Ubms1s/Ub,Npfixs1s,'--',color='cornflowerblue',linewidth=10)
    plt.plot(Ubms033s/Ub,Npfixs033s,'--',color='blue',linewidth=10)  
    #plt.plot(Ubms1/Ub,new_Npfixs1,'k:')
    #plt.plot(Ubms033/Ub,new_Npfixs033,'k:')
    plt.axhline(y=1,color='black',linewidth=1)
    
    plt.scatter(S085['6'][0:-4]/Ub,S085['9'][0:-4],color='cornflowerblue',s=700,marker='s',label=r"$s'=s_{b}$")
    plt.scatter(S03['6'][0:-2]/Ub,S03['9'][0:-2],color='blue',s=700,marker='s',label=r"$s'=0.3 \cdot s_{b}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=44); plt.yticks(fontsize=44)
    #plt.axhline(y=1,color='black',linewidth=3)
    #plt.legend(frameon=False,prop={'size': 48})
    plt.tight_layout();
    #plt.savefig("C:/Users/19084/Downloads/Ferrare_Good_Figure_4b.png",bbox_inches=0,dpi=900)
    plt.savefig("figures/figure_4b.png",bbox_inches='tight',dpi=700)
    #plt.show()

if __name__=='__main__':
    run()
