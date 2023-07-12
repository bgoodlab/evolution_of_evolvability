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
    #simulated parameters
    Ub=.00001
    N=10**8
    sb=.01
    sm=.03
    sm1=sm
    

    #import data
    sb1=pd.read_csv('data/Perturbative Results Exp/lbs_perturb_exp_Beneficial_Modifier_85203.0.csv')
    sb10=pd.read_csv('data/Perturbative Results Beta/lbs_perturb_beta_Beneficial_Modifier_85203.0.csv')
    Ubms=np.logspace(-6,-4,500)
    Ubms10=np.logspace(-10.2,-7,500)

    #simulated v for exponential and shorttailed distribution
    v10=.0000281
    v1=.000574
    xc1=fsolve_eq_xc_general(.01,.00001,.000574,.1,1,1)
    xc10=fsolve_eq_xc_general(.01,.00001,.0000281,.04,1,10)
    xcm10=fsolve_eq_xc_general(.03,Ub,v10,xc10,1,10)

    new_xc,new_seff,new_Ueff = calculate_xc_seff_Ueff_beta(v10,sb,Ub,10)
    

    #effective parameters
    s10_guess=.01
    delta_guess=((1/v10)+((10*(10-1))/(s10_guess**2))*(s10_guess/.01)**(10-2))**(-0.5)
    s1=xc1-v1/sb
    s10,delta=fsolve(equations_both_s_star_delta,[.01,delta_guess],args=(xc10,v10,.01,10))
    Ubeff1=Ub*short_tailed_exponential(s1,sb,1)*(2*np.pi*v1)**0.5
    Ubeff10=Ub*short_tailed_exponential(s10,sb,10)*(2*np.pi*delta**2)**0.5

    #print("Params:", xc10,new_xc,s10, new_seff, Ubeff10, new_Ueff)


    #theoretical predictions
    old_Npfixs1=[]
    old_Npfixs10=[]
    for Ubm in Ubms:
        #test
        xcm1=fsolve_eq_xc_exact_p_alt(N,s1,Ubeff1,sm,Ubm,v1,xc1,1,xc1)
        #test
        xcm1=fsolve_eq_xc_exact_p(N,s1,Ubeff1,sm,Ubm,v1,xc1,1)
        dxc=xcm1-xc1
        Npfix=(xcm1/xc1)*np.exp(-xc1*dxc/v1-dxc**2/(2*v1))
        old_Npfixs1.append(Npfix)
    for Ubm10 in Ubms10:
        xcm10=fsolve_eq_xc_exact_p(N,s10,Ubeff10,sm,Ubm10,v10,xc10,1)
        dxc=xcm10-xc10
        Npfix=(xcm10/xc10)*np.exp(-xc10*dxc/v10-dxc**2/(2*v10))
        old_Npfixs10.append(Npfix)
    old_Npfixs1=np.asarray(old_Npfixs1)
    old_Npfixs10=np.asarray(old_Npfixs10)

	######
	#
	# New theory calculation code
	#
	#-----
    Npfixs1 = []
    Npfixs10 = []
    for U1s, outputs, beta, v in zip([Ubms,Ubms10],[Npfixs1, Npfixs10],[1,10],[v1,v10]):
        for Ubm in U1s:
            Npfix = calculate_Npfix_beta_plus_delta(N,sb,Ub,beta,sm,Ubm,v)
            outputs.append(Npfix)
    #------
    #######
    
    
    #plot theory and simulation
    plt.figure(figsize=(14,12)); plt.xlabel("$U_{b}^{1}/U_{0}$",fontsize=65,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(\mu \to \mu + \delta \mu)$",fontsize=65,labelpad=15); plt.yscale('log'); plt.xscale('log')
    plt.plot(Ubms/Ub,Npfixs1,color='blue',linewidth=10)
    
    plt.scatter(sb1['6'][0:-3]/Ub,sb1['9'][0:-3],color='blue',s=700,marker='s',label=r"$\beta=1$")
    plt.scatter(sb10['6'][0:-9]/Ub,sb10['9'][0:-9],color='grey',s=700,marker='^',label=r"$\beta=10$")
    plt.plot(Ubms10/Ub,Npfixs10,color='grey',linewidth=10)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=44); plt.yticks(fontsize=48)
    plt.axhline(y=1,color='black',linewidth=1)
    plt.legend(frameon=False,prop={'size': 50})
    plt.tight_layout();
    plt.savefig("figures/figure_4c.png",bbox_inches='tight',dpi=700)
    #plt.show()

if __name__=='__main__':
    run()
