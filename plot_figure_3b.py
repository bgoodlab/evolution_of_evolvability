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
import sys

def run():
    #import simulation data
    Sr1=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb8521.0.csv')
    Sr09=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb85209.0.csv')
    Sr07=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb85207.0.csv')
    Sr05=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb85205.0.csv')
    Sr0=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb8520.0.csv')
    plt.figure(figsize=(14,12)); plt.xlabel("Direct benefit $s_{m}/s_{b}$", fontsize=65,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(\mu \to \mu_{m};s_{m})$",fontsize=65,labelpad=15); plt.yscale('log')

    #plot simulation data
    plt.scatter(np.abs(Sr1['9'])/.01,Sr1['10'],color='darkgreen',label='$s_{b}^{\'}/s_{b}=1$',s=300)
    plt.scatter(np.abs(Sr09['9'])/.01,Sr09['10'],color='seagreen',label='$s_{b}^{\'}/s_{b}=0.9$',s=300)
    plt.scatter(np.abs(Sr07['9'])/.01,Sr07['10'],color='mediumseagreen',label='$s_{b}^{\'}/s_{b}=0.7$',s=300)
    #plt.scatter(np.abs(Sr05['9'])/.01,Sr05['10'],color='b',label='$r_{s}=0.5$')
    plt.scatter(np.abs(Sr0['9'][1:])/.01,Sr0['10'][1:],color='mediumaquamarine',label='$s_{b}^{\'}/s_{b}=0$',s=300)
    
    #import simulation v
    sds=np.array(np.abs(Sr09['9']))
    v1=np.array(Sr1['0'])[-1]
    v09=np.array(Sr09['0'])[-1]
    v07=np.array(Sr07['0'])[-1]
    v0=np.array(Sr0['0'])[-1]
   
    # Base params
    N = 1e08
    Ub = 1e-05
    sb = 1e-02
    
    sds1=np.linspace(0.0000001,.04,100)
    sds09=np.linspace(0.0015,.04,100)
    sds07=np.linspace(0.0107,.04,100)
    sds0=np.linspace(0.0123,.04,100)

    #Theoretical predictions
    old_Tr1=Npfix_ben(.01,1,.00001,100000000,(-1)*sds1,v1)
    old_Tr09=Npfix_ben(.01,.9,.00001,100000000,(-1)*sds09,v09)
    old_Tr07=Npfix_ben(.01,.7,.00001,100000000,(-1)*sds07,v07)
    old_Tr0=Npfix_ben(.01,0,.00001,100000000,(-1)*sds0,v0)

    Tr1 = []
    Tr09 = []
    Tr07 = []
    Tr0 = []
    for deltams, outputs, v, rs in zip([sds1, sds09, sds07, sds0],[Tr1, Tr09, Tr07, Tr0],[v1,v09,v07,v0],[1,0.9,0.7,0]):
        
        Um = Ub
        sm = sb*rs
        
        for deltam in deltams:
            Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um,deltam=deltam)
            #print(N,sb,Ub,v,sm,Um,deltam,Npfix)
            outputs.append(Npfix)

    #plot theoretical predictions
    plt.plot(sds1/.01,Tr1,color='darkgreen',linewidth=4)
    plt.plot(sds09/.01,Tr09,color='seagreen',linewidth=4)
    plt.plot(sds07/.01,Tr07,color='mediumseagreen',linewidth=4)
    plt.plot(sds0/.01,Tr0,color='mediumaquamarine',linewidth=4)
    #plt.axhline(y=1,color='black',linestyle='--')
    
    
    #plt.plot(sds1/.01,old_Tr1,'k:')
    #plt.plot(sds09/.01,old_Tr09,'k:')
    #plt.plot(sds07/.01,old_Tr07,'k:')
    #plt.plot(sds0/.01,old_Tr0,'k:')     
    

    plt.axvline(x=0,color='black')
    plt.axhline(y=1,color='black',linewidth=1)
    plt.xticks([0,1,2,3,4],[0,1,2,3,4])
    plt.xticks(fontsize=40); plt.yticks(fontsize=40)
    plt.legend(frameon=False,loc='upper left',prop={'size': 36})
    plt.xticks(fontsize=40); plt.yticks(fontsize=40)
    plt.tight_layout();
    plt.savefig("figures/figure_3b.png",bbox_inches='tight',dpi=700)
    #plt.show()
    
if __name__=='__main__':
    run()
