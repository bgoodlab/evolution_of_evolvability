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

#Generate Figure 5top:
def run():
    #import simulation files
    S=pd.read_csv('data/Beneficial Modifier with Deleterious Mutations/WOODSPAPER4_24752.0.csv')
    S2=pd.read_csv('data/Beneficial Modifier with Deleterious Mutations/WOODSPAPER4_24_r_12_.0.csv')

    plt.figure(figsize=(10,2.7)); plt.ylabel("Mean fitness effect",fontsize=20,labelpad=15); plt.xscale('log');

    #rate of adaptation measured in simulation
    v=0.000333863


    #generate theory
    sb = 4e-02
    Ub = 1e-06
    N=10**8
    Ud=.0004
    sm=6e-02
    sm2=4.8e-02
    sd=-.1
    sbar=(Ub*sb+Ud*sd)/(Ub+Ud)

    #plot simulations
    #plt.scatter(S['8'],S['10'],label="$s_{b}'/s_{b}=1.5$",color='red',s=300)
    #plt.scatter(S2['8'],S2['10'],label="$s_{b}'/s_{b}=1.2$",color='tomato',s=300)

    s_bars=[]
    Udms=np.logspace(-3.5,-1.45,250)
    for Udm in Udms:
        s_bars.append((Ub*sm+Udm*sd)/(Ub+Udm))
    s_bars=np.asarray(s_bars)
    s_bars2=[]
    Udms2=np.logspace(-3.5,-1.45,250)
    for Udm in Udms2:
        s_bars2.append((Ub*sm2+Udm*sd)/(Ub+Udm))
    s_bars2=np.asarray(s_bars2)

    
    plt.plot(Udms,s_bars,label="$s_{b}'/s_{b}=1.5$",color='tomato',linewidth=3)
    plt.plot(Udms2,s_bars2,label="$s_{b}'/s_{b}=1.2$",color='red',linewidth=3)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #plt.axhline(y=1,color='black',linewidth=3)
    #plt.legend(frameon=True,prop={'size': 22})
    #plt.ylim(0,2)
    plt.yticks([0,-0.2])
    #plt.yscale('log')

    plt.tight_layout()
    #plt.show()
    plt.savefig('figures/figure_5b_top.png',bbox_inches='tight',dpi=900)

    
if __name__=='__main__':
    run()
