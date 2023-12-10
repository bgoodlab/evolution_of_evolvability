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
    Sk0=pd.read_csv('data/SI/SI_5_85200092.0.csv')
    Sk1=pd.read_csv('data/SI/SI_5_85200192.0.csv')
    Sk2=pd.read_csv('data/SI/SI_5_85200292.0.csv')
    Sk3=pd.read_csv('data/SI/SI_5_85200392.0.csv')
    Sk4=pd.read_csv('data/SI/SI_5_85200492.0.csv')
    Sk5=pd.read_csv('data/SI/SI_5_85200592.0.csv')
    Sk10=pd.read_csv('data/SI/SI_5_852001091.0.csv')

    plt.figure(figsize=(10,12)); plt.xlabel("$s_{b}'/s_{b}$",fontsize=50,labelpad=15); plt.ylabel(r"$ \tilde{p}_{fix}(s_{b} \to s_{b}' \to \emptyset)$",fontsize=50,labelpad=15); plt.yscale('log');

    #rate of adaptation measured in simulation
    v=0.0000496323251430607

    #generate theory
    sb = 1e-02
    Ub = 1e-05
    N=10**8

    #plot simulations
    plt.scatter(Sk10['5']/sb,Sk10['9'],color='darkred',label="$k=10$",s=300)
    plt.scatter(Sk5['5']/sb,Sk5['9'],color='red',label="$k=6$",s=300)
    plt.scatter(Sk4['5']/sb,Sk4['9'],color='orangered',label="$k=5$",s=300)
    plt.scatter(Sk3['5']/sb,Sk3['9'],color='tomato',label="$k=4$",s=300)
    plt.scatter(Sk2['5']/sb,Sk2['9'],color='salmon',label="$k=3$",s=300)
    plt.scatter(Sk1['5']/sb,Sk1['9'],color='lightsalmon',label="$k=2$",s=300)
    plt.scatter(Sk0['5'][1:]/sb,Sk0['9'][1:],color='peachpuff',label="$k=1$",s=300)

    Npfixs=[]
    sms=np.linspace(sb,6*sb,250)
    for sm in sms:
        Npfix= calculate_Npfix_twoparam(N,sb,Ub,v,sm,Ub,deltam=0)
        Npfixs.append(Npfix)

    
    plt.plot(sms/sb,np.asarray(Npfixs),color='black',linewidth=3)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.axhline(y=1,color='black',linewidth=3)
    plt.legend(frameon=True,prop={'size': 24})

    plt.tight_layout()
    plt.savefig('figures/figure_S3b.png',bbox_inches='tight',dpi=900)

    
if __name__=='__main__':
    run()

