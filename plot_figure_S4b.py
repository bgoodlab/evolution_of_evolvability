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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)

#Generate Figure 2b:
def run():
    #import simulation files

    
    S=pd.read_csv('data/SI/SI_6_X_decay.csv')
    S1=S[0:800]
    S2=S[4:27]

    fig,ax1=plt.subplots(figsize=(10,12)); ax1.set_xlabel("Generations, t",fontsize=50,labelpad=15); ax1.set_ylabel(r"$ Fitness, X (\%)$",fontsize=50,labelpad=15);


    #generate theory
    sb = 1e-02
    Ub = 1e-05
    N=10**7

    #plot simulations
    ax1.errorbar(S1['0'][::45],S1['11'][::45],S1['12'][::45],color='green',ls='none',linewidth=7,zorder=3)
    ax1.errorbar(S1['0'][1],S1['11'][1],S1['12'][1],color='green',ls='none',linewidth=7,zorder=3)
    ax1.errorbar(S1['0'][4],S1['11'][4],S1['12'][4],color='green',ls='none',linewidth=7,zorder=3)
    ax1.plot(S1['0'],S1['11'],color='black',linewidth=4,zorder=1)
    ax1.scatter(S1['0'][::45],S1['11'][::45],color='green',s=300,zorder=10)
    ax1.scatter(S1['0'][1],S1['11'][1],color='green',s=300,zorder=10)
    ax1.scatter(S1['0'][4],S1['11'][4],color='green',s=300,zorder=10)
    
    
    #lables=["0",r"$10^{1}$",r"$10^{2}$",r"$10^{3}$",r"$10^{4}$",r"$10^{5}$"]
    ax1.tick_params(axis='x',labelsize=32)
    ax1.tick_params(axis='y',labelsize=32)
    #ax1.set_xticklabels(lables)

    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.4,0.2,0.5,0.5])
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=2, loc2=3, fc="none", ec='0.5')
    ax2.errorbar(S2['0'][::4],S2['11'][::4],S2['12'][::4],color='green',ls='none',linewidth=3,zorder=3)
    ax2.plot(S2['0'][::4],S2['11'][::4],color='black',linewidth=3,zorder=3)
    ax2.scatter(S2['0'][::4],S2['11'][::4],color='green',s=200,zorder=10)
    ax2.tick_params(axis='x',labelsize=30)
    ax2.tick_params(axis='y',labelsize=30)
    
    #plt.axhline(y=1,color='black',linewidth=3)
    #plt.legend(frameon=False,loc='best',bbox_to_anchor=(0.5, 0., 0.5, 0.5),prop={'size': 36})
    #plt.tight_layout()
    #plt.show()
    plt.savefig('figures/figure_S4b.png',bbox_inches='tight',dpi=900)

    
if __name__=='__main__':
    run()
