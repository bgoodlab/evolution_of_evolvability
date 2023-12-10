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

#Generate Figure 2b:
def run():
    #import simulation files
    SUd05=pd.read_csv('data/SI/SI_1_v385201005.0.csv')
    SUd01=pd.read_csv('data/SI/SI_1_v385201001.0.csv')
    SUd0=pd.read_csv('data/SI/SI_1_v385201000.0.csv')

    plt.figure(figsize=(10,12)); plt.xlabel("$|s_{d}|/s_{b}$",fontsize=50,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(U_{d} \to U_{d}')$",fontsize=50,labelpad=15); plt.yscale('log');

    #rate of adaptation measured in simulation
    v=0.0000496323251430607

    #generate theory
    sb = 1e-02
    Ub = 1e-05
    N=10**8

    #plot simulations
    plt.scatter(SUd0['7']/-sb,SUd0['9'],color='r',label="$U_{d}'=0$",s=300)
    plt.scatter(SUd01['7']/-sb,SUd01['9'],color='g',label="$U_{d}'=0.1\cdot U_{d}$",s=300)
    plt.scatter(SUd05['7']/-sb,SUd05['9'],color='b',label="$U_{d}'=0.5\cdot U_{d}$",s=300)


    asymptote_0= calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub,deltam=.01)
    asymptote_01=calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub,deltam=.01-.1*.01)
    asymptote_05=calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub,deltam=.01-.5*.01)
    asymptote_0_001=calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub,deltam=.001)
    xc=calculate_xc_twoparam(v,sb,Ub)

    plt.axhline(y=asymptote_0,color='r',linewidth=3)
    plt.axhline(y=asymptote_01,color='g',linewidth=3)
    plt.axhline(y=asymptote_05,color='b',linewidth=3)
    #plt.axhline(y=asymptote_0_001,linestyle='--',color='r',linewidth=3)
    plt.axvline(x=v/(xc*sb),linestyle='--',color='black',linewidth=3)
    plt.axvline(x=0,color='black',linewidth=3)
    plt.xticks([1.5,1,0.5,0],fontsize=36)
    plt.yticks(fontsize=36)
    plt.axhline(y=1,color='black',linewidth=3)
    plt.legend(frameon=False,loc='best',bbox_to_anchor=(0.5, 0., 0.5, 0.5),prop={'size': 36})
    plt.tight_layout()
    plt.savefig('figures/figure_S2b.png',bbox_inches='tight',dpi=900)

    
if __name__=='__main__':
    run()
