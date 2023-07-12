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

def run():
    #import data
    S752=pd.read_csv('data/Mutation Rate Modifier No Deleterious/r_mutationrate_752.0.csv')
    S852=pd.read_csv('data/Mutation Rate Modifier No Deleterious/r_mutationrate_852.0.csv')
    S952=pd.read_csv('data/Mutation Rate Modifier No Deleterious/r_mutationrate_952.0.csv')

    Ub = 1e-05
    sb = 1e-02
    
    #plot data
    plt.figure(figsize=(14,12)); plt.xlabel("$U_{b}'/U_{b}$",fontsize=65,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(U_{b} \to U_{b}')$",fontsize=65,labelpad=15); plt.yscale('log'); plt.xscale('log')
    #plt.title('Theory vs. Simulation: DFE Modifiers; s=.01; Ub=1e-05')
    plt.scatter(S752['11'],S752['10'],color='r',label='$N=10^{7}$',s=300)
    plt.scatter(S852['11'],S852['10'],color='g',label='$N=10^{8}$',s=300)
    plt.scatter(S952['11'],S952['10'],color='b',label='$N=10^{9}$',s=300)
    
    #fsolve iterative guesses
    rs=np.array(S952['11'])[0:-1]
    q7=2*np.log(10000000*.01)/(np.log(.01/.00001))
    q8=2*np.log(100000000*.01)/(np.log(.01/.00001))
    q9=2*np.log(1000000000*.01)/(np.log(.01/.00001))

    #simulation v
    rs_smooth=np.linspace(1,100,100)
    rs_smooth_N7=np.linspace(1,90,100)
    v752=np.array(S752['0'])[-1]
    v852=np.array(S852['0'])[-1]
    v952=np.array(S952['0'])[-1]
    
      
    T752 = []
    T852 = []
    T952 = []      
    for N,v,outputs in zip([1e07,1e08,1e09],[v752,v852,v952],[T752,T852,T952]):
        Npfixs = []
        for rU in rs_smooth:
            Um = rU*Ub
            sm = sb
            xc = calculate_xc_twoparam(v752,sb,Ub)
            xcm = calculate_xc_twoparam(v752,sm,Um)
            Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um)
            #print(xc,xcm,Npfix)
            outputs.append(Npfix)
        
        
    
        
    #theoretical predictions
    old_T752=Npfix_mutation_rate(.01,rs_smooth,.00001,10000000,v752)
    old_T852=Npfix_mutation_rate(.01,rs_smooth,.00001,100000000,v852)
    old_T952=Npfix_mutation_rate(.01,rs_smooth,.00001,1000000000,v952)


    #plot theory
    plt.plot(rs_smooth,T752,color='r',linewidth=5) #theory
    plt.plot(rs_smooth,T852,color='g',linewidth=5)
    plt.plot(rs_smooth,T952,color='b',linewidth=5)
    
    
    plt.xticks(fontsize=48); plt.yticks(fontsize=48)
    plt.axhline(y=1,color='black',linewidth=1)
    plt.plot(rs,rs,color='black',linestyle='--',linewidth=5)
    plt.tight_layout();
    plt.legend(frameon=False,prop={'size': 48}); 
    plt.savefig("figures/figure_2c.png",bbox_inches='tight',dpi=700)
    
if __name__=='__main__':
    run()
    
