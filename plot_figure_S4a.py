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

    
    S=pd.read_csv('data/SI/SI_6_s_decay.csv')
    S_un=pd.read_csv('data/SI/SI_6_s_un_decay.csv')
    S1=S[0:800]
    S2=S[3:27]
    S1_un1=S_un[38:800]
    S2_un2=S_un[38:85]
    f=1.31595

    fig,ax1=plt.subplots(figsize=(10,12)); ax1.set_xlabel("Generations, t",fontsize=50,labelpad=15); ax1.set_ylabel(r"$ s_{b}(\overline{X})/s_{b}(t=0)$",fontsize=50,labelpad=15); ax1.set_yscale('log')


    #generate theory
    sb = 1e-02
    Ub = 1e-05
    N=10**7

    #plot simulations
    ax1.errorbar(S1['0'][::45],S1['11'][::45],S1['12'][::45],color='green',ls='none',linewidth=4,zorder=3)
    ax1.errorbar(S1['0'][1],S1['11'][1],S1['12'][1],color='green',ls='none',linewidth=4,zorder=3)
    ax1.errorbar(S1['0'][4],S1['11'][4],S1['12'][4],color='green',ls='none',linewidth=4,zorder=3)
    ax1.plot(S1['0'],S1['11'],color='black',linewidth=4,zorder=1)
    ax1.scatter(S1['0'][::45],S1['11'][::45],color='green',s=300,zorder=10)
    ax1.scatter(S1['0'][1],S1['11'][1],color='green',s=300,zorder=10)
    ax1.scatter(S1['0'][4],S1['11'][4],color='green',s=300,zorder=10)

    #ax1.errorbar(S1_un1['0'][::15],S1_un1['11'][::15]*0.6/f,S1_un1['12'][::15]*0.6,color='blue',ls='none',linewidth=4,zorder=3)
    #ax1.errorbar(S1_un1['0'][1],S1_un1['11'][1],S1_un1['12'][1],color='blue',ls='none',linewidth=4,zorder=3)
    #ax1.errorbar(S1_un1['0'][4],S1_un1['11'][4],S1_un1['12'][4],color='blue',ls='none',linewidth=4,zorder=3)
    #ax1.plot(S1_un1['0'],S1_un1['11']*0.6/f,color='black',linewidth=2,zorder=1)
    #ax1.scatter(S1_un1['0'][::15],S1_un1['11'][::15]*0.6/f,color='blue',s=60,zorder=10)
    #ax1.scatter(S1_un1['0'][1],S1_un1['11'][1],color='blue',s=60,zorder=10)
    #ax1.scatter(S1_un1['0'][4],S1_un1['11'][4],color='blue',s=60,zorder=10)
    ax1.plot(S1['0'][3::],.25*np.ones(len(S1['0'][3::])),color='blue',linestyle='--',linewidth=3)

    
    
    
    lables=[r"$0$",r"$10^{4}$",r"$2 \cdot 10^{4}$",r"$3 \cdot 10^{4}$",r"$4 \cdot 10^{4}$"]
    ax1.tick_params(axis='x',labelsize=36)
    ax1.tick_params(axis='y',labelsize=36)
    #ax1.ticklabel_format(axis='x', style='sci',scilimits=(0, 5))
    #ax1.set_xticklabels(lables)

    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.45,0.485,0.5,0.5])
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=2, loc2=3, fc="none", ec='0.5')
    ax2.errorbar(S2['0'][::4],S2['11'][::4],S2['12'][::4],color='green',ls='none',linewidth=3,zorder=3)
    ax2.plot(S2['0'][::4],S2['11'][::4],color='black',linewidth=3,zorder=3)
    ax2.scatter(S2['0'][::4],S2['11'][::4],color='green',s=100,zorder=10)
    ax2.plot(np.linspace(200,1150,500),.25*np.ones(500),color='blue',linestyle='--',linewidth=3)

    #ax2.errorbar(S2_un2['0'][::10],S2_un2['11'][::10]*0.6/f,S2_un2['12'][::10]*0.6/f,color='blue',ls='none',linewidth=3,zorder=3)
    #ax2.plot(S2_un2['0'][::10],S2_un2['11'][::10]*0.6/f,color='black',linestyle='--',linewidth=3,zorder=3)
    #ax2.scatter(S2_un2['0'][::10],S2_un2['11'][::10]*0.6/f,color='blue',s=100,zorder=10)
    
    ax2.tick_params(axis='x',labelsize=28)
    ax2.tick_params(axis='y',labelsize=30)
    ax2.set_yscale('log')
    
    #plt.axhline(y=1,color='black',linewidth=3)
    #plt.legend(frameon=False,loc='best',bbox_to_anchor=(0.5, 0., 0.5, 0.5),prop={'size': 36})
    #plt.tight_layout()
    plt.savefig('figures/figure_S4a.png',bbox_inches='tight',dpi=900)

    
if __name__=='__main__':
    run()
