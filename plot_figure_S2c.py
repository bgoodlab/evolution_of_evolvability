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
    S=pd.read_csv('data/SI/SI_4_new_8520101.0.csv')

    plt.figure(figsize=(10,12)); plt.xlabel("$U_{d}$",fontsize=50,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(U_{d} \to 0)$",fontsize=50,labelpad=15); plt.yscale('log'); plt.xscale('log');

    #rate of adaptation measured in simulation
    v=0.0000496323251430607

    #generate theory
    sb = 1e-02
    Ub = 1e-05
    N=10**8

    #plot simulations
    plt.scatter(S['4'][2::],S['9'][2::],color='grey',s=300)

    Npfixs=[]
    Uds=np.logspace(np.log10(.02),np.log10(.00005),250)
    for Ud in Uds:
        S_t= calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub,deltam=Ud)
        Npfixs.append(S_t)
    Npfixs=np.asarray(Npfixs)
    
    plt.plot(Uds,Npfixs,color='grey',linewidth=3,label=r"$s_{d}=-10^{-2}$")
    plt.axhline(y=1,color='black',linewidth=3)
    plt.axvline(x=.01,color='black',linestyle='--',linewidth=3)
    plt.legend(frameon=False,loc='best',prop={'size': 30})
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.tight_layout()

    plt.savefig('figures/figure_S2c.png',bbox_inches='tight',dpi=900)

    
if __name__=='__main__':
    run()
