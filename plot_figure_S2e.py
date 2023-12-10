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
    SUd01=pd.read_csv('data/SI/SI_38520101.0.csv')
    SUd001=pd.read_csv('data/SI/SI_385200101.0.csv')

    plt.figure(figsize=(10,12)); plt.xlabel("$\Delta U_{b}=-\Delta U_{d}$",fontsize=50,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(U_{b},U_{d} \to U_{b}',U_{d}')$",fontsize=50,labelpad=15); plt.yscale('log'); plt.xscale('log')

    #rate of adaptation measured in simulation
    v=0.0000496323251430607

    #generate theory
    sb = 1e-02
    Ub = 1e-05
    N=10**8

    #plot simulations
    #plt.scatter((SUd01['6']-SUd01['2'])/Ub,SUd01['9'],color='b',s=300,label="$U_{d}=.01$")
    plt.scatter((SUd001['6']-SUd001['2']),SUd001['9'],color='b',s=300,label="$U_{d}=.001$")

    DUd01= np.logspace(-5,-3,10)
    DUd001= np.logspace(-5,-3,10)
    
    
    Npfixs01=[]
    Npfixs001=[]
    for D in DUd01:
        Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub+D,deltam=D)
        Npfixs01.append(Npfix)
    for D in DUd001:
        Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub+D,deltam=D)
        Npfixs001.append(Npfix)

    #plt.plot(DUd01/Ub,np.asarray(Npfixs01),color='b',linewidth=3)
    plt.plot(DUd001,np.asarray(Npfixs001),color='b',linewidth=3)

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.axhline(y=1,color='black',linewidth=3)
    #plt.legend(frameon=False,prop={'size': 36})

    plt.tight_layout()
    plt.savefig('figures/figure_S2e.png',bbox_inches='tight',dpi=900)
    
if __name__=='__main__':
    run()
