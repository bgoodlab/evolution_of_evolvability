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
import seaborn as sea
import matplotlib.colors as colors

#Figure S5
def run():

    #plt.figure(figsize=(10,12)); plt.xlabel(r"$ U_{b}'/U_{b}$",fontsize=50,labelpad=15); plt.ylabel(r"$s_{b}'/s_{b}$", fontsize=50,labelpad=15); plt.xscale('log');

    
    # Base params
    N = 1e08
    sb = 1e-02
    Ub = 1e-05
    rUs=np.logspace(-np.log10(20),np.log10(20),30)
    rss=np.linspace(4,0.75,30)
    X,Y = np.meshgrid(rUs,rss)
    sd=-.00990001
    sb=.009900001
    
    values=[]
    gammas=[]
    g=[]
        
    for rs in rss:
        gammas1=[]
        for rU in rUs:
            values.append([rU,rs])
            Npfix_no_cost = calculate_Npfix_twoparam_v_unknown(N,sb,Ub,rs*sb,rU*Ub,deltam=0)
            if Npfix_no_cost>=1:
                Npfix_cost=calculate_Npfix_twoparam_v_unknown(N,sb,Ub,rs*sb,rU*Ub,deltam=sd)
                Npfix_short=calculate_Npfix_twoparam_v_unknown(N,sb,Ub,sb,Ub,deltam=sd)
            else:
                Npfix_cost=calculate_Npfix_twoparam_v_unknown(N,sb,Ub,rs*sb,rU*Ub,deltam=sb)
                Npfix_short=calculate_Npfix_twoparam_v_unknown(N,sb,Ub,sb,Ub,deltam=sb)
            gamma=(np.log(Npfix_cost)-np.log(Npfix_no_cost))/np.log(Npfix_short)
            if Npfix_no_cost >0:
                gamma=(np.log(Npfix_cost)-np.log(Npfix_no_cost))/np.log(Npfix_short)
            else:
                gamma=10
            #print ("rU= ",rU,"rs= ",rs,"gamma= ",gamma,"Modifier w/ benefit",Npfix_cost,"Direct Benefit ",Npfix_short," Ordinary Npfix ",Npfix_no_cost,gamma)
            gammas1.append(gamma)
        gammas.append(gammas1)
    gammas=np.asarray(gammas)
    
    #plot theoretical predictions
    #plt.contourf(X,Y,gammas)
    divnorm=colors.TwoSlopeNorm(vmin=0.3, vcenter=1, vmax=2)
    sea.set(font_scale=3)
    hm=sea.heatmap(gammas,xticklabels=False,yticklabels=rss,cmap='seismic',norm=divnorm)
    #plt.figure(figsize=(10,12));
    plt.xlabel(r"$ U_{b}'/U_{b}$",fontsize=50,labelpad=15); plt.ylabel(r"$s_{b}'/s_{b}$", fontsize=50,labelpad=15); #plt.xscale('log');
    


    #plot (previous) theoretical predictions
    #plt.plot(-sds12/.01,old_Tr12,'k:')
    #plt.plot(-sds17/.01,old_Tr17,'k:')
    #plt.plot(-sds22/.01,old_Tr22,'k:')
    #plt.plot(sds40/.01,old_Tr40,'k:')
    #plt.plot(sds60/.01,old_Tr60,'k:')
    
    #plt.legend(frameon=True,prop={'size': 32})
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    #plt.axvline(x=1,color='black',linewidth=3)
    #plt.axhline(y=1,color='black',linewidth=3)
    plt.tight_layout();
    plt.show()
    plt.savefig("figures/figure_S5.png",bbox_inches='tight',dpi=700)
    
if __name__=='__main__':
    run()

