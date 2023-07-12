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
    #simulation parameters
    N=1e08
    Ub=1e-05
    sb=1e-02
    sm10=.03
    sm1=.08618
    
    Ubm10 = 4.2*1e-08
    Ubm1 = .00000024793
    
    #simulation data
    s1_s=pd.read_csv('data/Perturbative Direct Cost Exp/lbs_perturb_exp_Beneficial_Modifier_newMAY_dc85208618.0.csv')
    s10_s=pd.read_csv('data/Perturbative Direct Cost Beta/lbs_perturb_beta_Beneficial_Modifier_MAY_dc852012.0.csv')
    xc1=fsolve_eq_xc_general(.01,.00001,.000574,.1,1,1)

    #effective parameters
    v10=.0000281
    v1=.000574
    s1=xc1-v1/.01
    s10_guess=.01
    delta_guess=((1/v10)+((10*(10-1))/(s10_guess**2))*(s10_guess/.01)**(10-2))**(-0.5)
    xc10=fsolve_eq_xc_general(.01,Ub,v10,.03,1,10)
    s10,delta=fsolve(equations_both_s_star_delta,[.01,delta_guess],args=(xc10,v10,.01,10))
    Ub10=Ub*short_tailed_exponential(s10,.01,10)*(2*np.pi*delta**2)**0.5
    Ubeff1=Ub*short_tailed_exponential(s1,.01,1)*(2*np.pi*v1)**0.5

    
    sds=np.linspace(-.000001,-.03,250)
    sds10=np.linspace(-.00000001,-.02,250)

    xcm1=fsolve_eq_xc_exact_p(N,s1,Ubeff1,sm1,.00000024793,v1,xc1,1)
    xcm10=fsolve_eq_xc_exact_p(N,s10,Ub10,sm10,4.2*10**(-8),v10,xc10,1)
    
    #Npfix beneficial modifier with direct cost
    def Npfix_bmdm(d,sb,sm1,N,v,xc,xcm):
        xcm=xcm
        k=np.floor(np.abs(d)/sb)
        D=np.abs(d)-k*sb
        Npfix=(xcm*sm1/(xc*sb))*np.exp((xc**2-xcm**2)/(2*v))*np.exp(-(xcm-sm1/2)*np.abs(d)/v)*np.exp(-D**2/(2*v)+sm1*D/(2*v))/factorial(k)*((1-sm1/xcm)**k)*((1-np.exp(-D*np.minimum(sm1,sm1-D+xc-xcm)/v))/(sm1*D/v)+((xcm-sm1)/xcm)/(k+1)*((1-np.exp(-sm1*(sm1-D)/v))/((sm1*(sm1-D)/v))))
        return Npfix

    #Npfix beneficial modifier with direct cost
    def Npfix_bmdm10(sds,s,sm10,v,xc,xcm):
        outs=[]
        for sd in sds10:
            r=sm10/s
            sm=r*s
            Ub=4.2*10**(-8)
            dxc=xcm-xc
            k=np.floor(np.abs(sd)/(sm10))
            D=np.abs(sd)-k*r*s
            Npfix_nm=(xcm/(xc))*np.exp(-dxc*xc/v-dxc**2/(2*v))
            
            def integrand(x,r,s,sd,v,k):
                return (Ub/sm)**k*(v/(sm*xcm*np.sqrt(2*np.pi*v)))*np.exp(-(x-sd)**2/(2*v))*((x+(k+1)*sm)/sm)*gamma(-x/sm-k)/gamma(1-x/sm) +(Ub/sm)**k*(1/factorial(k))*np.exp(-k*sm*D/v-D**2/(2*v)-x*D/v)*0.5*(erf((x+k*sm)/np.sqrt(2*v))+1)*(1/xcm)+(v/(sm*np.sqrt(2*np.pi*v)))*np.exp(-(x-sd)**2/(2*v))*(1/factorial(k))*(sm/(x+k*sm))*(1/xcm)*(Ub/sm)**k
            def integrand_extra(x,r,s,sd,v,k):
                return (Ub/sm)**(k+1)*(np.exp(-k*sm*D/v-D**2/(2*v)+(2*k+1)*sm**2/(2*v)))*(1/factorial(k+1))*np.exp(x*(sm-D)/v)*0.5*(erf((x+(k+1)*sm)/np.sqrt(2*v))+1)*(1/xcm)
            val=integrate.quad(integrand,xcm-(k+1)*sm,np.minimum(xcm-k*sm,xc-k*sm-D),args=(r,s,sd,v,k))
            val_e=integrate.quad(integrand_extra,xcm-(k+2)*sm,xcm-(k+1)*sm,args=(r,s,sd,v,k))
            Npfix=Npfix_nm*(val[0]+val_e[0])
            
            outs.append(Npfix)
        Npfixvals=np.asarray(outs)
        return Npfixvals
    old_Npfixs1=Npfix_bmdm(sds,s1,sm1,N,v1,xc1,xcm1)
    old_Npfixs10=Npfix_bmdm10(sds,s10,sm10,v10,xc10,xcm10)
    
    ######
	#
	# New theory calculation code
	#
	#-----
    Npfixs1 = []
    Npfixs10 = []
    for deltams,outputs,v,sm,Ubm,beta in zip([sds,sds10],[Npfixs1, Npfixs10],[v1,v10],[sm1,sm10],[Ubm1,Ubm10],[1,10]):
        for deltam in deltams:
            Npfix = calculate_Npfix_beta_plus_delta(N,sb,Ub,beta,sm,Ubm,v,deltam=deltam)
            outputs.append(Npfix)
    #-----
    #######
    
    
    #plot simulation and theory
    plt.figure(figsize=(14,12)); plt.xlabel("Direct cost $s_{m}/s_{0}$",fontsize=65); plt.ylabel(r"$ \tilde{p}_{fix}(\mu \to \mu +\delta \mu;s_{m})$",fontsize=65);  
    plt.scatter(s1_s['10']/.01,s1_s['9'],color='cornflowerblue',s=700,marker='s')
    plt.scatter(s10_s['10']/.01,s10_s['9'],color='grey',marker='^',s=700)
    plt.plot(sds/.01,Npfixs1,color='cornflowerblue',linewidth=10)
    plt.plot(sds10/.01,Npfixs10,color='grey',linewidth=10)
    # Old version
    #plt.plot(sds/sb,old_Npfixs1,'r:')
    #plt.plot(sds10/sb,old_Npfixs10,'r:')
    #print("s1=",s1,"s10=",s10,"Ub10=",Ub10, 'xcm10=',xcm10, )
    plt.axhline(y=1,color='black',linewidth=1)
    plt.yscale('log')
    plt.axvline(color='black',linewidth=1)
    plt.xticks([0,-1,-2,-3],[0,-1,-2,-3])
    plt.xticks(fontsize=44); plt.yticks(fontsize=44)
    #plt.legend(frameon=False,prop={'size': 32})
    plt.tight_layout();
    plt.savefig("figures/figure_4d.png",bbox_inches='tight',dpi=700)
    #plt.show()
    
if __name__=='__main__':
    run()
