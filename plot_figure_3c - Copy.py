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
    plt.figure(figsize=(14,12)); plt.xlabel(r"$ \log \tilde{p}_{fix}(\mu \to \mu_{m})$", fontsize=58,labelpad=15); plt.ylabel(r"$ \log \tilde{p}_{fix}(s_{m}|\mu)$",fontsize=58,labelpad=15);

    #import simulation data
    Sr12=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85212.0.csv')
    Sr13=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85213.0.csv')
    Sr14=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85214.0.csv')
    Sr15=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85215.0.csv')
    Sr17=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85217.0.csv')

    Sr19=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85219.0.csv')
    Sr22=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85222.0.csv')
    Sr25=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85225.0.csv')
    Sr40=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85240.0.csv')
    Sr60=pd.read_csv('data/Modifier on Deleterious Mutation/s_d85260.0.csv')

    Sr19_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85219.0.csv')
    Sr22_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85222.0.csv')
    Sr25_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85225.0.csv')
    Sr30_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85230.0.csv')
    Sr35_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85235.0.csv')
    Sr40_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85240.0.csv')
    Sr50_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85250.0.csv')
    Sr60_c=pd.read_csv('data/Modifier on Deleterious Mutation/s_d_cross85260.0.csv')

    Sr09=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb85209.0.csv')
    Sr085=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sbNEW_cross852085.0.csv')
    Sr08=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sbNEW_cross85208.0.csv')
    Sr075=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sbNEW_cross85208.0.csv')
    Sr07=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb85207.0.csv')
    Sr065=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sbNEW_cross852065.0.csv')
    Sr06=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sbNEW_cross85206.0.csv')
    Sr055=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sbNEW_cross852055.0.csv')
    Sr05=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb85205.0.csv')
    Sr0=pd.read_csv('data/Deleterious Modifier w Beneficial Mutation/sb8520.0.csv')


    #approximate Npfix near Npfix=1 with linear function for deleterious modifier with direct benefit
    Sr09fit=np.polyfit(np.abs(Sr09['9'][0:2]), np.log(Sr09['10'][0:2]), 1)
    Sr085fit=np.polyfit(np.abs(Sr085['9'][0:4]), np.log(Sr085['10'][0:4]), 1)
    Sr08fit=np.polyfit(np.abs(Sr08['9'][0:4]), np.log(Sr08['10'][0:4]), 1)
    Sr075fit=np.polyfit(np.abs(Sr075['9'][0:4]), np.log(Sr075['10'][0:4]), 1)
    Sr07fit=np.polyfit(np.abs(Sr07['9'][0:4]), np.log(Sr07['10'][0:4]), 1)
    Sr065fit=np.polyfit(np.abs(Sr065['9'][0:4]), np.log(Sr065['10'][0:4]), 1)
    Sr06fit=np.polyfit(np.abs(Sr06['9'][0:5]), np.log(Sr06['10'][0:5]), 1)
    Sr055fit=np.polyfit(np.abs(Sr055['9'][0:5]), np.log(Sr055['10'][0:5]), 1)
    Sr05fit=np.polyfit(np.abs(Sr05['9'][0:4]), np.log(Sr05['10'][0:4]), 1)
    Sr0fit=np.polyfit(np.abs(Sr0['9'][0:4]), np.log(Sr0['10'][0:4]), 1)

    #direct benefit that makes modifier neutral
    Sr09zero=-Sr09fit[1]/Sr09fit[0]
    Sr085zero=-Sr085fit[1]/Sr085fit[0]
    Sr08zero=-Sr08fit[1]/Sr08fit[0]
    Sr075zero=-Sr075fit[1]/Sr075fit[0]
    Sr07zero=-Sr07fit[1]/Sr07fit[0]
    Sr065zero=-Sr065fit[1]/Sr065fit[0]
    Sr06zero=-Sr06fit[1]/Sr06fit[0]
    Sr055zero=-Sr055fit[1]/Sr055fit[0]
    Sr05zero=-Sr05fit[1]/Sr05fit[0]
    Sr0zero=-Sr0fit[1]/Sr0fit[0]

    sims_rs_2=np.asarray([.9,0.85,0.8,0.75,.7,0.65,0.6])#,0.55,.5])
    zeros_del=np.asarray([Sr09zero,Sr085zero,Sr08zero,Sr075zero,Sr07zero,Sr065zero,Sr06zero])#,Sr055zero,Sr05zero])    

    #approximate Npfix near Npfix=1 with linear function for beneficial modifier with direct cost
    Sr12fit=np.polyfit(np.abs(Sr12['9'][0:4]), np.log(Sr12['10'][0:4]), 1)
    Sr13fit=np.polyfit(np.abs(Sr13['9'][0:4]), np.log(Sr13['10'][0:4]), 1)
    Sr14fit=np.polyfit(np.abs(Sr14['9'][0:4]), np.log(Sr14['10'][0:4]), 1)
    Sr15fit=np.polyfit(np.abs(Sr15['9'][0:4]), np.log(Sr15['10'][0:4]), 1)
    Sr17fit=np.polyfit(np.abs(Sr17['9'][0:2]), np.log(Sr17['10'][0:2]), 1)
    Sr19fit=np.polyfit(np.abs(Sr19['9'][0:3]), np.log(Sr19['10'][0:3]), 1)
    Sr22fit=np.polyfit(np.abs(Sr22['9'][0:4]), np.log(Sr22['10'][0:4]), 1)
    Sr25fit=np.polyfit(np.abs(Sr25['9'][0:4]), np.log(Sr25['10'][0:4]), 1)
    Sr30fit=np.polyfit(np.abs(Sr30_c['9'][0:4]), np.log(Sr30_c['10'][0:4]), 1)
    Sr35fit=np.polyfit(np.abs(Sr35_c['9'][0:2]), np.log(Sr35_c['10'][0:2]), 1)
    Sr40fit=np.polyfit(np.abs(Sr40['9'][0:2]), np.log(Sr40['10'][0:2]), 1)
    Sr50fit=np.polyfit(np.abs(Sr50_c['9'][0:2]), np.log(Sr50_c['10'][0:2]), 1)
    Sr60fit=np.polyfit(np.abs(Sr60['9'][0:2]), np.log(Sr60['10'][0:2]), 1)

    #direct costs that make modifier neutral
    Sr12zero=-Sr12fit[1]/Sr12fit[0]
    Sr13zero=-Sr13fit[1]/Sr13fit[0]
    Sr14zero=-Sr14fit[1]/Sr14fit[0]
    Sr15zero=-Sr15fit[1]/Sr15fit[0]
    Sr17zero=-Sr17fit[1]/Sr17fit[0]
    Sr19zero=-Sr19fit[1]/Sr19fit[0]
    Sr22zero=-Sr22fit[1]/Sr22fit[0]
    Sr25zero=-Sr25fit[1]/Sr25fit[0]
    Sr30zero=-Sr30fit[1]/Sr30fit[0]
    Sr35zero=-Sr35fit[1]/Sr35fit[0]
    Sr40zero=-Sr40fit[1]/Sr40fit[0]
    Sr50zero=-Sr50fit[1]/Sr50fit[0]
    Sr60zero=-Sr60fit[1]/Sr60fit[0]

    #direct costs and benefits that make modifier neutral
    zeros=np.asarray([Sr12zero,Sr13zero,Sr14zero,Sr15zero,Sr17zero,Sr19zero,Sr22zero,Sr25zero,Sr30zero,Sr35zero,Sr40zero,Sr50zero])#,Sr60zero])
    sim_rs=np.asarray([1.2,1.3,1.4,1.5,1.7,1.9,2.2,2.5,3.0,3.5,4.0,5])#,6]

    #fixation probability of beneficial mutation
    def Npfix_Beneficial_Mutation(s,sb,N,Ub,v):
            q=sb*np.log(N*sb)/np.log(sb/Ub)
            xc=fsolve_eq_xc_exact(N,sb,Ub,v,q*sb,1)
            Npfix=(v/(sb*s))*np.exp(-s**2/(2*v)+xc*s/v)*(1-np.exp(-sb*s/v))+N*s*(erf(xc/np.sqrt(2*v))-erf((xc-s)/np.sqrt(2*v))) #+ (v/(xc*sb))*(np.exp(xc*s/v-s**2/(2*v))-1)
            return Npfix

    #fixation probability of deleterious mutation
    def Npfix_Deleterious_Mutation(d,sb,N,Ub,v):
        k=np.floor(np.abs(d)/sb)
        D=np.abs(d)-k*sb
        q=sb*np.log(N*sb)/np.log(sb/Ub)
        xc=fsolve_eq_xc_exact(N,sb,Ub,v,q*sb,1)
        #Npfix=(np.exp(-d**2/(2*v))/gamma(k+1))*((1-sb/xc)**k)*np.exp(-k*(k-1)*sb**2/(2*v))*(np.exp(-(xc-(k+1)*sb)*D/v-k*sb*(xc-k*sb)/v)*((1-np.exp(-D*(sb-D)/v))/(sb*D/v))+(np.exp(-k*sb**2/v)/(k+1))*((xc-sb)/xc)*np.exp(-(xc-(k+1)*sb)*(k*sb+D)/v)*((1-np.exp(-sb*(sb-D)/v))/(sb*(sb-D)/v)))
        Npfix=np.exp(-(xc-sb/2)*np.abs(d)/v)*np.exp(-D**2/(2*v)+sb*D/(2*v))/factorial(k)*(1-sb/xc)**k*((1-np.exp(-D*(sb-D)/v))/(sb*D/v)*(1-np.exp(-sb*(sb-D)/v))+((xc-sb)/xc)/(k+1)*(1-np.exp(-sb*(sb-D)/v))/((sb*(sb-D)/v))*(1-np.exp(-D*sb/v)))
        return Npfix

    def Npfix_r(s,rs,Ub,N,v):
        out=[]
        for r in rs:
            q_it=2*np.log(N*s)/(np.log(s/Ub))
            xc=fsolve_eq_xc_exact(N,s,Ub,v,q_it*s,1)
            xcm=fsolve_eq_xc_exact(N,s,Ub,v,xc,r)
            Npfix2=2*N*r*Ub*xcm*s/v
            Npfix1=(r*xcm/xc)*np.exp((xc**2-xcm**2)/(2*v))
            Npfix=np.minimum(Npfix1,Npfix2)
            out.append(Npfix)
        return np.asarray(out)
    def Npfix_r_2(s,rs,Ub,N,v):
        out=[]
        for r in rs:
            q_it=2*np.log(N*s)/(np.log(s/Ub))
            xc=fsolve_eq_xc_exact(N,s,Ub,v,q_it*s,1)
            xcm=fsolve_eq_xc_exact(N,s,Ub,v,xc,r)
            Npfix2=2*N*r*Ub*xcm*s/v
            Npfix1=(r*xcm/xc)*np.exp((xc**2-xcm**2)/(2*v))
            Npfix=np.minimum(Npfix1,Npfix2)
            out.append(Npfix)
        return np.asarray(out)

    #fixation probability of beneficial modifier with direct cost in quasi-sweeps regime
    def Npfix_del_constrain_large_r(sd,s,r,Ub,N,v):
        sd=sd[0]
        Npfix=0
        q=2*np.log(N*s)/np.log(s/Ub)
        xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
        q=xc/s
        sd=-sd
        sm=r*s
        xcm1=2*s*(np.log(N*s)**0.5/np.log(s/Ub))*(np.log(s/(Ub*r)))**0.5
        xcm=fsolve_eq_xc_exact(100000000,r*s,Ub,v,xc,1)
        k=np.floor(np.abs(sd)/(sm))
        D=np.abs(sd)-k*r*s
        Npfix_nm=2*N*r*Ub*xcm*s/v
        sd=-sd
        def integrand(x,r,s,sd,v,k):
            return (Ub/sm)**k*(v/(sm*xcm*np.sqrt(2*np.pi*v)))*np.exp(-(x-sd)**2/(2*v))*((x+(k+1)*sm)/sm)*gamma(-x/sm-k)/gamma(1-x/sm) +(Ub/sm)**k*(1/factorial(k))*np.exp(-k*sm*D/v-D**2/(2*v)-x*D/v)*0.5*(erf((x+k*sm)/np.sqrt(2*v))+1)*(1/xcm)+(v/(sm*np.sqrt(2*np.pi*v)))*np.exp(-(x-sd)**2/(2*v))*(1/factorial(k))*(sm/(x+k*sm))*(1/xcm)*(Ub/sm)**k
        val=integrate.quad(integrand,xcm-(k+1)*sm,np.minimum(xcm-k*sm,xc-k*sm-D),args=(r,s,sd,v,k))
        def integrand_extra(x,r,s,sd,v,k):
            return (Ub/sm)**(k+1)*(np.exp(-k*sm*D/v-D**2/(2*v)+(2*k+1)*sm**2/(2*v)))*(1/factorial(k+1))*np.exp(x*(sm-D)/v)*0.5*(erf((x+(k+1)*sm)/np.sqrt(2*v))+1)*(1/xcm)
        val_e=integrate.quad(integrand_extra,xcm-(k+2)*sm,xcm-(k+1)*sm,args=(r,s,sd,v,k))

        Npfix=Npfix_nm*(val[0]+val_e[0])
        return np.log(Npfix)

    #fixation probability of beneficial modifier with direct cost in multiple mutations regime
    def Npfix_del_constrain_small_r(sd,s,r,Ub,N,v):
        sd=sd[0]
        Npfix=0
        q=2*np.log(N*s)/np.log(s/Ub)
        xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
        sm=r*s
        xcm=fsolve_eq_xc_exact(100000000,r*s,Ub,v,xc,1)
        k=np.floor(np.abs(sd)/(sm))
        D=np.abs(sd)-k*r*s
        sd=-sd
        Npfix=(xcm*sm/(xc*s))*np.exp((xc**2-xcm**2)/(2*v))*np.exp(-(xcm-sm/2)*np.abs(sd)/v)*np.exp(-D**2/(2*v)+sm*D/(2*v))/factorial(k)*((1-sm/xcm)**k)*((1-np.exp(-D*np.minimum(sm,sm-D+xc-xcm)/v))/(sm*D/v)*(1-np.exp(-sm*(sm-D)/v))+((xcm-sm)/xcm)/(k+1)*((1-np.exp(-sm*(sm-D)/v))/((sm*(sm-D)/v))*(1-np.exp(-D*sm/v))))
        return np.log(Npfix)

    #Fixation probability of deleterious modifier with direct benefit
    def Npfix_ben_constrain(sd,s,r,Ub,N,v):
        out=[]
        sd=-sd
        q=(2*np.log(N*s)/(np.log(s/Ub)))
        xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
        q=xc/s
        xcm1=1/r*(xc+s/2*(r**2-1)+(v/s)*np.log(r))
        xcm=fsolve_eq_xc_exact(N,r*s,Ub,v,xc,1)
        if 0<r and r<(2*q)**0.5:
            if xc-sd>xcm:
                Npfix=((v*xcm)/(s*sd*xc))*np.exp((xc**2-(xcm+sd)**2)/(2*v))*(np.exp(r*s*sd/v)-1) + (v/(xc*s))*(np.exp((xc**2-(xcm+sd)**2)/(2*v))-1)+N*(sd)*(erf((xcm+sd)/(2*v)**0.5)-erf(xc/np.sqrt(2*v)))
            else:
                Npfix=-((v*xcm/(s*sd*xc)))*np.exp((xc**2-xcm**2-sd**2)/(2*v))*(np.exp(-sd*(-sd+xc)/v)-np.exp(-(xcm-r*s)*sd/v))
        return np.log(Npfix)

    #function to solve for direct cost that makes benefical modifier neutral  
    def solve_for_zero(s,r,v,Ub,N):
        xc=fsolve_eq_xc_exact(N,s,Ub,v,.03,1)
        xcm=fsolve_eq_xc_exact(N,r*s,Ub,v,.03,1)
        Npfix1=(r*xcm/xc)*np.exp((xc**2-xcm**2)/(2*v))
        Npfix2=Npfix_nm=2*N*r*Ub*xcm*s/v
        if r*s<xcm:
            guess=r*s+.00000001
            root=fsolve(Npfix_del_constrain_small_r,[guess],args=(s,r,Ub,N,v),epsfcn=.0005)
        else:
            guess=r*s+.00000001
            root=fsolve(Npfix_del_constrain_large_r,[guess],args=(s,r,Ub,N,v),epsfcn=.0005)
        return root

    #function to solve for direct benefit that makes deleterious modifier neutral
    def solve_for_zero_del(s,r,v,Ub,N):
        guess=(1/r)*s+.00000001
        root=fsolve(Npfix_ben_constrain,[guess],args=(s,r,Ub,N,v))
        return root

    #solve for zeros for direct cost
    Srzerostheory=[]

    rs_vals=np.linspace(1,4.5,350)
    for r in rs_vals:
        Srzerostheory.append(solve_for_zero(.01,r,4.957205463377411e-05,.00001,100000000)[0])
    Srzerostheory=np.asarray(Srzerostheory)

    #solve for zeros for direct benefit
    Srzerosdel=[]
    theory_rs_2=np.linspace(.6,1,350)
    for r in theory_rs_2:
        Srzerosdel.append(solve_for_zero_del(.01,r,4.957205463377411e-05,.00001,100000000)[0])
    Srzerosdel=np.asarray(Srzerosdel)
    for i in range(0,len(Srzerosdel)):
        xc=fsolve_eq_xc_exact(10**8,.01,.00001,4.957205463377411e-05,.03,1)
        v=4.957205463377411e-05
        s=.01
        e=v/xc*np.log(xc*s/v*xc**2/v)
        xcm=np.sqrt(2)*xc*(1-e/(2*xc))
        scrit=np.sqrt(2)*(1-e/(2*xc))*xc-xc
        if Srzerosdel[i]>scrit:
            Srzerosdel[i]=scrit

    #new

    N=1e08
    sb=1e-02
    Ub=1e-05
    v=4.957205463377411e-05
    new_theory_rs = np.linspace(0.5,4.5,100)
    critical_costs = []
    critical_cost_Npfixs = []
    critical_modifier_Npfixs = []
    for rs in new_theory_rs:
        Um=Ub
        sm=sb*rs

        deltam_critical = calculate_critical_cost_twoparam(N,sb,Ub,v,sm,Um)
        critical_costs.append(deltam_critical)

        # Calculate the Npfix for direct cost alone
        cost_Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sb,Ub,deltam=deltam_critical)
        critical_cost_Npfixs.append(cost_Npfix)

        # Calculate the Npfix for evolvability part alone
        modifier_Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um,deltam=0,correct_xcm=False)
        critical_modifier_Npfixs.append(modifier_Npfix)


        print("Critical cost for sm=", rs,deltam_critical,cost_Npfix, modifier_Npfix)

    critical_cost_Npfixs = np.array(critical_cost_Npfixs)
    critical_modifier_Npfixs = np.array(critical_modifier_Npfixs)

    critical_xs = np.log(critical_modifier_Npfixs)
    critical_ys = np.log(critical_cost_Npfixs)

    #simulation
    Npfixsb=np.log(Npfix_Beneficial_Mutation(.01,.01,10**8,.00001,.0000495))
    Npfix_modben=Npfix_r(.01,sim_rs,.00001,10**8,.0000495)
    Npfix_moddel=Npfix_r_2(.01,sims_rs_2,.00001,10**8,.0000495)
    Npfix_ben=Npfix_Beneficial_Mutation(zeros_del,.01,10**8,.00001,.0000495)
    Npfix_del=Npfix_Deleterious_Mutation(-1*zeros,.01,10**8,.00001,.0000495)

    #theory
    Npfix_modben_t=Npfix_r(.01,rs_vals,.00001,10**8,.0000495)
    Npfix_moddel_t=Npfix_r_2(.01,theory_rs_2,.00001,10**8,.0000495)
    Npfix_ben_t=Npfix_Beneficial_Mutation(Srzerosdel,.01,10**8,.00001,.0000495)
    Npfix_del_t=Npfix_Deleterious_Mutation(-1*Srzerostheory,.01,10**8,.00001,.0000495)

    #Npfix_modben_t=np.asarray([Npfix_modben_t[i][0] for i in range(0,len(Npfix_modben_t))])
    #Npfix_moddel_t=np.asarray([Npfix_moddel_t[i][0] for i in range(0,len(Npfix_moddel_t))])
    Npfix_modben_t=np.asarray(Npfix_modben_t)
    Npfix_moddel_t=np.asarray(Npfix_moddel_t)


    xmin = -2.75*Npfixsb
    #xmax = np.log(critical_modifier_Npfixs[-1])
    xmax = 2*Npfixsb
    ymin = -8*Npfixsb
    ymax = -xmin

    # GET RID OF THE SCALING ON THE X/Y AXES
    Npfixsb=1

    #plot sim
    plt.scatter(np.log(Npfix_modben)/Npfixsb,np.log(Npfix_del)/Npfixsb,color='black',s=300)
    plt.scatter(np.log(Npfix_moddel)/Npfixsb,np.log(Npfix_ben)/Npfixsb,color='black',s=300)
    plt.scatter([0],[0],color='black')

    #plot theory
    # Old version
    #plt.plot(np.log(Npfix_modben_t)/Npfixsb,np.log(Npfix_del_t)/Npfixsb,"r:")
    #plt.plot(np.log(Npfix_moddel_t)/Npfixsb,np.log(Npfix_ben_t)/Npfixsb,'r:')

    plt.plot(critical_xs/Npfixsb, critical_ys/Npfixsb,color='black',linewidth=7)


    #plot features
    plt.axhline(color='black',linewidth=5)
    plt.axvline(color='black',linewidth=5)
    plt.plot(np.linspace(xmin,xmax,100)/Npfixsb, -1*np.linspace(xmin,xmax,100)/Npfixsb,linestyle='--',color='black',linewidth=4,label="Additive expectation")
    plt.fill_between(critical_xs/Npfixsb, critical_ys/Npfixsb,ymax/Npfixsb*np.ones_like(critical_ys),color='honeydew',zorder=0)
    plt.fill_between([critical_xs[-1]/Npfixsb,xmax/Npfixsb], [ymin/Npfixsb,ymin/Npfixsb],[ymax/Npfixsb,ymax/Npfixsb],color='honeydew',zorder=0)
    plt.fill_between(critical_xs/Npfixsb,ymin/Npfixsb*np.ones_like(critical_ys),critical_ys/Npfixsb,color='aliceblue',zorder=0)

    plt.xticks(fontsize=44); plt.yticks(fontsize=44)
    #plt.legend(frameon=False,prop={'size': 40})
    plt.gca().set_xlim([xmin/Npfixsb,xmax/Npfixsb])
    plt.gca().set_ylim([ymin/Npfixsb,ymax/Npfixsb])

    plt.tight_layout();
    #plt.savefig("figures/figure_3c.png",bbox_inches='tight',dpi=700)
    plt.show()

if __name__=='__main__':
    run()
