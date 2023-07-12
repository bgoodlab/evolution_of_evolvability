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

#Figure 3a
def run():
    #import simulation data
    Sr12=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85212.0.csv')
    Sr17=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85217.0.csv')
    Sr22=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85222.0.csv')
    Sr40=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85240.0.csv')
    Sr60=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85260.0.csv')

    plt.figure(figsize=(14,12)); plt.xlabel("Direct cost $s_{m}/s_{b}$", fontsize=65,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(\mu \to \mu_{m};s_{m})$",fontsize=65,labelpad=15); plt.yscale('log')

    #plot simulation data
    plt.scatter(-np.abs(Sr12['9'][1:-1])/.01,Sr12['10'][1:-1],color='mediumpurple',label="$s_{b}'/s_{b}=1.2$",s=300)
    plt.scatter(-np.abs(Sr17['9'])/.01,Sr17['10'],color='mediumslateblue',label="$s_{b}'/s_{b}=1.7$",s=300)
    plt.scatter(-np.abs(Sr22['9'])/.01,Sr22['10'],color='blue',label="$s_{b}'/s_{b}=2.2$",s=300)
    plt.scatter(-np.abs(Sr40['9'])/.01,Sr40['10'],color='darkslateblue',label="$s_{b}'/s_{b}=4.0$",s=300)
    plt.scatter(-np.abs(Sr60['9'])/.01,Sr60['10'],color='midnightblue',label="$s_{b}'/s_{b}=6.0$",s=300)

    #import v from simulation
    sds=np.array(np.abs(Sr12['9']))
    v12=np.array(Sr12['0'])[-1]
    v17=np.array(Sr17['0'])[-1]
    v22=np.array(Sr22['0'])[-1]
    v40=np.array(Sr40['0'])[-2]
    v60=np.array(Sr60['0'])[-1]
    
    
    # Base params
    N = 1e08
    sb = 1e-02
    Ub = 1e-05
    
    # SDS to scan through
    # (warning: for historical reasons, some have minus signs and some don't. 
    #   This accounted for below.)  
    sds12=np.linspace(0.0000000001,1.08*sb,300)
    sds17=np.linspace(0.00000000001,1.6999*sb,300)
    sds22=np.linspace(0.00000000001,2.25*sb,300)
    sds40=np.linspace(-0.00000000001,-5*sb,300)
    sds60=np.linspace(-0.00000000001,-7.4*sb,300)
    
    #Previous theoretical predictions for Npfix
    old_Tr12=Npfix_del(sb,1.2,Ub,N,sds12,v12)
    old_Tr17=Npfix_del(sb,1.7,Ub,N,sds17,v17)
    old_Tr22=Npfix_del(sb,2.2,Ub,N,sds22,v22)
    old_Tr40=Npfix_del(sb,4.0,Ub,N,sds40,v40)
    old_Tr60=Npfix_del(sb,6,Ub,N,sds60,v60)
    
    # New theoretical predictions for Npfix
    Tr12 = []
    Tr17 = []
    Tr22 = []
    Tr40 = []
    Tr60 = []
    
    for sds, outputs, v, rs in zip([-1*sds12,-1*sds17,-1*sds22, sds40, sds60],[Tr12, Tr17, Tr22, Tr40, Tr60],[v12,v17,v22,v40,v60],[1.2,1.7,2.2,4.0,6.0]):
        
        sm = sb*rs
        Um = Ub
        
        for sd in sds:
            Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um,deltam=sd)
            outputs.append(Npfix)
    
    #plot theoretical predictions
    plt.plot(-sds12/.01,Tr12,color='mediumpurple',linewidth=5)
    plt.plot(-sds17/.01,Tr17,color='mediumslateblue',linewidth=5)
    plt.plot(-sds22/.01,Tr22,color='blue',linewidth=5)
    plt.plot(sds40/.01,Tr40,color='darkslateblue',linewidth=5)
    plt.plot(sds60/.01,Tr60,color='midnightblue',linewidth=5)
    plt.axhline(y=1,color='black',linewidth=1)

    #plot (previous) theoretical predictions
    #plt.plot(-sds12/.01,old_Tr12,'k:')
    #plt.plot(-sds17/.01,old_Tr17,'k:')
    #plt.plot(-sds22/.01,old_Tr22,'k:')
    #plt.plot(sds40/.01,old_Tr40,'k:')
    #plt.plot(sds60/.01,old_Tr60,'k:')
    
    plt.legend(frameon=False,loc=0,prop={'size': 32})
    plt.xticks(fontsize=40); plt.yticks(fontsize=40)
    plt.axvline(x=0,color='black')
    plt.tight_layout();
    plt.savefig("figures/figure_3a.png",bbox_inches='tight',dpi=700)
    
if __name__=='__main__':
    run()
