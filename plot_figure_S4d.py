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
    S1=pd.read_csv('data/SI/SI_6_X_decay.csv')
    

    plt.figure(figsize=(10,12)); plt.xlabel("Mutation",fontsize=50,labelpad=15); plt.ylabel(r"$s_{b,j}'(X)/s_{b,j}(X)$", fontsize=50,labelpad=15);


    #generate theory
    sb = .025
    Ub = 1e-05
    N=10**7
    v=.00020848763

    X0_st=0.013210899

    s0=.025
    sigma=0.2
    X0=np.log(0.25)*(-sigma)
    t0=170
    rs=[]
    rs_95_l=[]
    rs_95_h=[]
    X=X0
    xc=calculate_xc_twoparam(v,sb,Ub)
    q=2*np.log(N*sb)/np.log(sb/Ub)
    steps=[]
    f0=0
    Xn=X0
    for i in range(0,int(np.ceil(q))):
        a=.1*np.exp(-(Xn)/sigma)
        steps.append(a)
        Xn+=a
    xc_value=0
    for i in range(0,int(np.floor(q))):
        xc_value+=steps[i]
    xc_value+=steps[int(np.floor(q))]*(q-np.floor(q))
    #xc=xc_value
        
        
    
    for i in range(0,4):
        f=np.exp(-(X+xc)/sigma)
        print(.025/(f*0.1))
        rs.append(0.25/f)
        #f_l=np.exp(-(X+xc-X0_st)/sigma)
        #rs_95_l.append(f_l*0.1)
        #f_h=np.exp(-(X+xc+X0_st)/sigma)
        #rs_95_h.append(f_h*0.1)
        X+=f*0.1
    rs=np.asarray(rs); rs_95_l=.01/np.asarray(rs_95_l); rs_95_h=.01/np.asarray(rs_95_h);
    #a=np.column_stack((rs_95_l,rs_95_h))
    #a=np.reshape(a,(2,4))
    rs=np.asarray(rs)
    rs=rs-rs[0]+1
    plt.bar(np.asarray([1,2,3,4]),rs,color='cornflowerblue',label=r"$\theta_{m}=\infty$")
 
    plt.axhline(y=1,color='green',linewidth=5,linestyle='--',label=r"$\theta_{m}=\theta$")
    #plt.legend(frameon=False,loc='best',bbox_to_anchor=(0.5, 0., 0.5, 0.5),prop={'size': 36})
    plt.xticks([1,2,3,4],fontsize=36)
    plt.ylim(0.8, 1.7)
    plt.yticks([1,1.25,1.5],fontsize=36)
    plt.legend(frameon=False,prop={'size': 36})
    plt.tight_layout()
    plt.savefig('figures/figure_S4d.png',bbox_inches='tight',dpi=900)

    
if __name__=='__main__':
    run()
