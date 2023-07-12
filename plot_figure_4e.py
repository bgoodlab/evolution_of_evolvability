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

    #import simulation parameters
    N=1e08
    Ub=1e-05
    sb=1e-02

    #import simulation data
    sb1=pd.read_csv('data/Exp Dead End Modifier/lbs_exp_Beneficial_Modifier_cost85201.0.csv')
    sb10=pd.read_csv('data/Beta Dead End Modifier/lbs_perturb_beta_Beneficial_Modifier_MAY_deadend_FINAL2852012.0.csv')
    sds1=np.linspace(.0475,.08,250)
    sds10=np.linspace(.0093,.019,250)

    #solve for effective parameters
    xc1=fsolve_eq_xc_general(.01,.00001,.000574,.1,1,1)
    xc10=fsolve_eq_xc_general(.01,.00001,.0000281,.04,1,10)
    v10=0.0000280322546307374
    v1=.000574
    s1=xc1-v1/.01
    s10_guess=.01
    delta_guess=((1/v10)+((10*(10-1))/(s10_guess**2))*(s10_guess/.01)**(10-2))**(-0.5)
    s10,delta=fsolve(equations_both_s_star_delta,[.01,delta_guess],args=(xc10,v10,.01,10))
    Ubeff1=Ub*short_tailed_exponential(s1,sb,1)*(2*np.pi*v1)**0.5
    Ub10=Ub*short_tailed_exponential(s10,sb,10)*(2*np.pi*delta**2)**0.5

    #xcm for dead-end
    eps1=v1/xc1*np.log(xc1*s1/v1*xc1**2/v1)
    eps10=v10/xc10*np.log(xc10*s10/v10*xc10**2/v10)
    xcm1=np.sqrt(2)*xc1*(1-eps1/(2*xc1))
    xcm10=np.sqrt(2)*xc10*(1-eps10/(2*xc10))

    #Npfix for dead-end
    def npfix_de(xcm,s,sd,v,xc):
        Npfix=(v/(xc*s))*(np.exp((xc**2-(xcm-sd)**2)/(2*v))-1) +N*(sd)*((1+erf((sd-xcm)/(2*v)**0.5))-(1+erf(-xc/np.sqrt(2*v))))
        return Npfix
    old_Npfixs10=npfix_de(xcm10,s10,sds10,v10,xc10)
    old_Npfixs1=npfix_de(xcm1,s1,sds1,v1,xc1)

	######
	#
	# New theory calculation code
	#
	#-----
    Npfixs1 = []
    Npfixs10 = []
    for deltams, outputs, v, beta in zip([sds1,sds10],[Npfixs1,Npfixs10],[v1,v10],[1,10]):
        for deltam in deltams:
            Ubm=0
            sm=sb
            Npfix = calculate_Npfix_beta(N,sb,Ub,beta,sm,Ubm,beta,v,deltam=deltam)
            outputs.append(Npfix)
    #-----
    ########
            
    #plot simulation and theory
    plt.figure(figsize=(14,12)); plt.xlabel("Direct benefit $s_{m}/s_{0}$",fontsize=65); plt.ylabel(r"$ \tilde{p}_{fix}(\mu \to \emptyset; s_{m})$",fontsize=65); plt.yscale('log'); 
    plt.scatter(sb1['10']/.01,sb1['9'],color='cornflowerblue',s=700,label='Beta=1',marker='s')
    plt.scatter(sb10['10'][0:-2]/.01,sb10['9'][0:-2],color='grey',s=700,label='Beta=10',marker='^')
    plt.plot(sds1/sb,Npfixs1,color='cornflowerblue',linewidth=10)
    plt.plot(sds10/sb,Npfixs10,color='grey',linewidth=10)
    
    #Old version
    #plt.plot(sds1/sb,old_Npfixs1,'r:')
    #plt.plot(sds10/sb,old_Npfixs10,'r:')
    
    
    #plt.axhline(y=1,linestyle='--',color='black')
    plt.yscale('log')
    plt.xticks(fontsize=40); plt.yticks(fontsize=40)
    plt.axhline(y=1,color='black',linewidth=1)
    plt.axvline(color='black',linewidth=1)
    #plt.legend(frameon=False,prop={'size': 32})
    plt.tight_layout();
    plt.savefig("figures/figure_4e.png",bbox_inches='tight',dpi=700)
    #plt.show()
    
if __name__=='__main__':
    run()
