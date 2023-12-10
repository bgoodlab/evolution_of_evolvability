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
    S=pd.read_csv('data/SI/SI_2_v385200101.0.csv')

    plt.figure(figsize=(10,12)); plt.xlabel("$s'/s$",fontsize=50,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(s_{b} \to s_{b}')$",fontsize=50,labelpad=15); plt.yscale('log');

    #rate of adaptation measured in simulation
    v=0.0000496323251430607

    #generate theory
    sb = 1e-02
    Ub = 1e-05
    N=10**8

    #plot simulations
    plt.scatter(S['5']/S['1'],S['9'],color='r',s=300)

    sms= np.linspace(.01,.034,200)
    
    Npfixs=[]
    for sm in sms:
        Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Ub,deltam=0)
        Npfixs.append(Npfix)

    plt.plot(sms/sb,np.asarray(Npfixs),color='r',linewidth=3)

    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.axhline(y=1,color='black',linewidth=3)
    

    plt.tight_layout()
    plt.savefig('figures/figure_S2d.png',bbox_inches='tight',dpi=900)
    
if __name__=='__main__':
    run()
