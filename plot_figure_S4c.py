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
from scipy.special import digamma

#Generate Figure 2b:


def run():
    #import simulation files
    S=pd.read_csv('data/SI/new_dr.0.csv')
    S1=pd.read_csv('data/SI/new_dr_2.csv')

    plt.figure(figsize=(10,12)); plt.xlabel("$s_{b}'(t_{0})/s_{b}(t_{0})$",fontsize=50,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(\tilde{s_{b}},\theta \to \tilde{s_{b}'},\theta_{m})$",fontsize=50,labelpad=15); plt.yscale('log'); plt.xscale('log')

    #rate of adaptation measured in simulation
    v=.00020848763

    #generate theory
    sb = .025
    Ub = 1e-05
    N=10**7
    theta=0.2
    shift=1.46667173
    t_est=(sb/shift)/0.000104843249755623
    v2=0.000104843249755623
    

    #plot simulations
    plt.scatter(S['5']*shift/.025,S['11'],color='b',label=r"$ \theta_{m}=\infty$",s=300)
    plt.scatter(S1['5']/.1,S1['10'],color='g',label=r"$\theta_{m}=\theta$",s=300)


    Npfixs=[]
    rs=np.linspace(1/shift,10,250)
    rs2=np.linspace(1,10,250)
    for r in rs2:
        Npfixs.append(calculate_Npfix_twoparam(N,sb,Ub,v,r*sb,Ub,deltam=0))
    Npfixs=np.asarray(Npfixs)
    xc=calculate_xc_twoparam(v,sb,Ub)
    q=xc/sb
    xc2=calculate_xc_twoparam(v2,sb,Ub)
    plt.plot(rs2,Npfixs,color='green',linewidth=3)

    Npfixs2=[]
    for r in rs:
        vv=[r*1.46667173,r*1.59715302,r*1.72717468,r*1.85680909]
        vv1=[1.46667173,1.59715302,1.72717468,1.85680909]
        #vv1=[1.4190109941306224,1.5496818748265315,1.6798615583656253,1.8096297712308993]
        #Npfix_alt=calculate_Npfix_twoparam(N,sb,Ub,v,r*sb,Ub,deltam=0)
        Npfix_alt=Npfix_large_r(sb,r,Ub,N,v)
        #v11=np.exp((sb/shift)*t_est*((sb*r)/(sb/vv1[0])-1))
        #v12=np.exp((sb/shift)*t_est*((2*sb*r)/(sb/vv1[0]+sb/vv1[1])-1))
        #v13=np.exp((sb/shift)*t_est*((3*sb*r)/(sb/vv1[0]+sb/vv1[1]+sb/vv1[2])-3))
        v11=np.exp((sb/shift)*t_est*((sb*r)/(sb/vv1[0])-1))
        v12=np.exp((sb/shift)*t_est*((sb*r)/(sb/vv1[0])+(sb*r)/(sb/vv1[1])-2))
        Npfix_val=v11*(v11*v12) #*(v11*v12*v13)
        Npfixs2.append(np.minimum(Npfix_val,Npfix_alt))
    plt.plot(rs,Npfixs2,color='blue',linewidth=4)

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.axhline(y=1,color='black',linewidth=3)
    plt.axvline(x=1,color='black',linewidth=3)
    plt.legend(frameon=False,loc='best',bbox_to_anchor=(0.5, 0., 0.5, 0.5),prop={'size': 40})
    plt.tight_layout()
    plt.savefig('figures/figure_S4c.png',bbox_inches='tight',dpi=900)

   
if __name__=='__main__':
    run()
