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

#Figure 2c
def run():
    #import data
    Srs12=pd.read_csv('data/Modifier Supply Effect/supply_effect_paper_85212.0.csv')
    Srs15=pd.read_csv('data/Modifier Supply Effect/supply_effect_paper_85215.0.csv')
    Srs20=pd.read_csv('data/Modifier Supply Effect/supply_effect_paper_85220.0.csv')

    plt.figure(figsize=(14,12)); plt.xlabel(r"$ \tilde{p}_{fix}(s_{b} \to s_{b}') \cdot \tilde{p}_{fix}(U_{b} \to  U_{b}') $", fontsize=65,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(U_{b},s_{b} \to U_{b}',s_{b}')$",fontsize=65,labelpad=15); plt.xscale('log'); plt.yscale('log')

    rs12=np.array(Srs12['11'])[0:-1]
    rs15=np.array(Srs15['11'])[0:-3]
    rs20=np.array(Srs20['11'])[2:-1]
    rs_smooth=np.linspace(1,10,250)

    v12=np.array(Srs12['0'])[-1]
    v15=np.array(Srs15['0'])[-1]
    v20=np.array(Srs20['0'])[-1]

    N = 1e08
    sb = 1e-02
    Ub = 1e-05
    
    Trs12 = []
    Trs15 = []
    Trs20 = []
    Trs12_w = []
    Trs15_w = []
    Trs20_w = []
    for rUs,rs,v,output,additive_output in zip([rs12,rs15,rs20],[1.2,1.5,2],[v12,v15,v20],[Trs12,Trs15,Trs20],[Trs12_w,Trs15_w,Trs20_w]):
        for rU in rUs:
            sm = sb*rs
            Um = Ub*rU
            Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um,correct_xcm=False)
            Npfix_s = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Ub,correct_xcm=False)
            Npfix_U = calculate_Npfix_twoparam(N,sb,Ub,v,sb,Um,correct_xcm=False)
            
            output.append(Npfix)
            additive_output.append(Npfix_s*Npfix_U)
            
    #True Theoretical Npfix theoretical prediction
    old_Trs12=Npfix_supply_effect_alt(.01,rs12,.00001,100000000,1.2,v12)
    old_Trs15=Npfix_supply_effect_alt(.01,rs15,.00001,100000000,1.5,v15)
    old_Trs20=Npfix_supply_effect_alt(.01,rs20,.00001,100000000,2,v20)
    
    #Additive Npfix theoretical prediction
    old_Trs12_w=Npfix_supply_effect_wrong_alt(.01,rs12,.00001,100000000,1.2,v12)
    old_Trs15_w=Npfix_supply_effect_wrong_alt(.01,rs15,.00001,100000000,1.5,v15)
    old_Trs20_w=Npfix_supply_effect_wrong_alt(.01,rs20,.00001,100000000,2,v20)
    
    #plot
    plt.plot(Trs12_w,Trs12,color='coral',linewidth=5)
    plt.plot(Trs15_w,Trs15,color='blue',linewidth=5)
    plt.plot(Trs20_w,Trs20,color='olive',linewidth=5)

    #boundary for grey region
    def boundary(xs,N,s):
        #s=.01
        #xc=fsolve_eq_xc_exact(10**8,.01,.00001,v12,.03,1)
        xc=calculate_xc_twoparam(v12,sb,Ub)
        a=(xc**2/(2*v12)-(xc*s)/(2*v12))
        #switches to large r at (s/U)^q-1 min=x^(1/q)*(NUb*log(s/Ub))^(1-1/q)
        return np.exp(np.log(xs)-np.log(xs)**2/(4*a))
        #return np.power(xs,(1-np.log(xs)/(4*np.log(N*s))))
    
    plt.axvline(x=1,color='black',linewidth=5)
    plt.axhline(y=1,color='black',linewidth=5)
    plt.plot(np.logspace(-7,7,100),np.logspace(-7,7,100),'--',color='black',linewidth=5) 
    plt.fill_between(np.logspace(-6,7,100), boundary(np.logspace(-6,7,100),100000000,.01),10**(-8),color='whitesmoke',zorder=0)
    
    #plot simulation data
    plt.scatter(Trs12_w,Srs12['10'][0:-1],color='coral',label='$s_{b}^{\'}/s_{b}=1.2$',s=300)
    plt.scatter(Trs15_w,Srs15['10'][0:-3],color='blue',label='$s_{b}^{\'}/s_{b}=1.5$',s=300)
    plt.scatter(Trs20_w,Srs20['10'][2:-1],color='olive',label='$s_{b}^{\'}/s_{b}=2$',s=300)

    plt.xticks(fontsize=38); plt.yticks(fontsize=38)
    plt.legend(frameon=False,loc=2,prop={'size': 40})
    
    # Set XLIM and YLIM
    plt.gca().set_xlim([1e-07,1e07])
    plt.gca().set_ylim([1e-07,1e07])
    plt.tight_layout();
    plt.savefig("figures/figure_2d.png",bbox_inches='tight',dpi=700)
    #plt.show()
    
if __name__=='__main__':
    run()
