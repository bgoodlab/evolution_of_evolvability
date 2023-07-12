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
    S752=pd.read_csv('data/Modifier No Deleterious Mutations/r_752.0.csv')
    S852=pd.read_csv('data/Modifier No Deleterious Mutations/r_852.0.csv')
    S952=pd.read_csv('data/Modifier No Deleterious Mutations/r_952.0.csv')

    plt.figure(figsize=(10,12)); plt.xlabel("$s_{b}'/s_{b}$",fontsize=65,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(s_{b} \to s_{b}')$",fontsize=65,labelpad=15); plt.yscale('log'); plt.xscale('log')

    #plot simulations
    plt.scatter(S752['11'],S752['10'],color='r',label='$N=10^{7}$',s=300)
    plt.scatter(S852['11'],S852['10'],color='g',label='$N=10^{8}$',s=300)
    plt.scatter(S952['11'],S952['10'],color='b',label='$N=10^{9}$',s=300)

    #rate of adaptation measured in simulation
    v752=np.array(S752['0'])[-1]
    v852=np.array(S852['0'])[-1]
    v952=np.array(S952['0'])[-1]
    rs=np.array(S952['11'])[0:-1]

    #SSWM Theory
    plt.plot(rs,rs,color='black',label='SSWM',linestyle='--',linewidth=5)

    #iterative predictions for xc/sb
    q7=2*np.log(10000000*.01)/(np.log(.01/.00001))
    q8=2*np.log(100000000*.01)/(np.log(.01/.00001))
    q9=2*np.log(1000000000*.01)/(np.log(.01/.00001))

    rs=np.linspace(1,10,300)

    #generate theory
    sb = 1e-02
    Ub = 1e-05
    
    T752=[]
    T852=[]
    T952=[]
    for i in range(0,len(rs)):
        Npfix_s=Npfix_small_r(.01,rs[i],.00001,10000000,v752)
        Npfix_l=Npfix_large_r(.01,rs[i],.00001,10000000,v752)
        Npfix = calculate_Npfix_twoparam(1e07,sb,Ub,v752,sb*rs[i],Ub)
        #T752.append(np.minimum(Npfix_s,Npfix_l))
        T752.append(Npfix)
        
    for i in range(0,len(rs)):
        Npfix_s=Npfix_small_r(.01,rs[i],.00001,100000000,v852)
        Npfix_l=Npfix_large_r(.01,rs[i],.00001,100000000,v852)
        Npfix = calculate_Npfix_twoparam(1e08,sb,Ub,v852,sb*rs[i],Ub)
        #T852.append(np.minimum(Npfix_s,Npfix_l))
        T852.append(Npfix)
      
    for i in range(0,len(rs)):
        Npfix_s=Npfix_small_r(.01,rs[i],.00001,1000000000,v952)
        Npfix_l=Npfix_large_r(.01,rs[i],.00001,1000000000,v952)
        Npfix = calculate_Npfix_twoparam(1e09,sb,Ub,v952,sb*rs[i],Ub)
        
        #T952.append(np.minimum(Npfix_s,Npfix_l))
        T952.append(Npfix)
        
    T752=np.asarray(T752); T852=np.asarray(T852); T952=np.asarray(T952)
    
    #plot theory predictions
    
    plt.plot(rs,T752,color='r',linewidth=5) #theory
    plt.plot(rs,T852,color='g',linewidth=5)
    plt.plot(rs,T952,color='b',linewidth=5)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.axhline(y=1,color='black',linewidth=1)

    plt.tight_layout()
    plt.savefig('figures/figure_2b.png',bbox_inches='tight',dpi=900)
    
if __name__=='__main__':
    run()
