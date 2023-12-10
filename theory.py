import numpy as np
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.special import erfc
from scipy.special import ndtr as gaussian_cdf
from scipy.special import gamma, gammaln
from scipy.stats import norm
from scipy import integrate
from scipy.special import factorial

def calculate_logNpfix_beneficial(s,xc,v,sb,Ub):

    #value = (v/s/sb)*(np.exp(xc**2/(2*v)-(xc-s)**2/(2*v)) - np.exp(-(s)**2/(2*v)) + np.exp(xc**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(s**2/xc)*2*gaussian_cdf((s-xc)/np.sqrt(v)))
    
    # Same version but factored out some of the big stuff
    
    value_without_xcsv = (v/s/sb)*(np.exp(-s*s/2/v)*(-1*np.expm1(-xc*s/v)) + np.exp(xc**2/(2*v)-xc*s/v)*(np.sqrt(np.pi/(2*v)))*(s**2/xc)*2*gaussian_cdf((s-xc)/np.sqrt(v)))
    if value_without_xcsv<0:
        print("Bad value:", s,xc,v,sb,Ub,value_without_xcsv)
    
    #return np.log(value)
    return xc*s/v+np.log(value_without_xcsv)
    
def constrain_xc_exact(xc,v,s,Ub):
    return calculate_logNpfix_beneficial(s,xc,v,s,Ub)+np.log(s*Ub/v)
    # previous version for records
    #return np.log((Ub/s)*(np.exp(xc**2/(2*v))*np.exp(-(xc-s)**2/(2*v)) - np.exp(-(s)**2/(2*v)) + np.exp(xc**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(s**2/xc)*(1+erf((s-xc)/np.sqrt(2*v))))) #correct#    
    
# previous version for solving for xc 
# (I've now defined a new one below that doesn't require N or xc_pop)
def fsolve_eq_xc_exact(N,s,Ub,v,xc_pop,r):
    #if r<2*(xc_pop/s)**0.5:
    xc=2*s*np.log(N*s)/np.log(s/Ub)
    if r<np.sqrt(2*xc*s):
        guess_xc=(1/r)*(xc_pop+(s/2)*(r**2-1))
    else:
        guess_xc=np.sqrt(2*v)*np.log(v/(Ub*r*s))**(0.5)
    root=fsolve(constrain_xc_exact,[guess_xc],args=(v,r*s,Ub))[0]
    return root

# new version (same thing, slightly different initial guesses, doesn't require N or xc_pop)
def calculate_xc_twoparam(v,sb,Ub):

    # Initial guesses for xc:
    # First one from MM regime
    guess_xc_mm = v/sb*np.log(sb/Ub)+sb/2
    # Second one from QS regime
    guess_xc_qs = np.sqrt(2*v*np.log(v/Ub/sb))
    #print(guess_xc_mm,guess_xc_qs)
    # Take larger of the two
    guess_xc = max([guess_xc_mm,guess_xc_qs])
    
    # Solve for xc numerically using starting guess
    condition = lambda x: constrain_xc_exact(x,v,sb,Ub)
    xc=fsolve(condition,guess_xc)[0]
    return xc

# new version (same thing, slightly different initial guesses, doesn't require N or xc_pop)
def calculate_both_twoparam(N,sb,Ub):

    # Initial guesses for xc:
    v=2*sb**2*np.log(N*sb)/np.log(sb/Ub)**2
    # First one from MM regime
    guess_xc_mm = v/sb*np.log(sb/Ub)+sb/2
    # Second one from QS regime
    guess_xc_qs = np.sqrt(2*v*np.log(v/Ub/sb))
    #print(guess_xc_mm,guess_xc_qs)
    # Take larger of the two
    guess_xc = max([guess_xc_mm,guess_xc_qs])
    
    # Solve for xc numerically using starting guess
    v,xc=fsolve_both(sb,Ub,N,v,guess_xc)
    return v,xc

#use both constraints to constrain both
def equations_both(p,s,Ub,N):
    v,xc=p
    return [constrain_xc_exact(xc,v,s,Ub),constrain_v(xc,v,N,s)]

#solve for xc and v for general population
def fsolve_both(s,Ub,N,guess_v,guess_xc):
    root=fsolve(equations_both,[guess_v,guess_xc],args=(s,Ub,N))
    return root

#v constrain equation
def constrain_v(xc,v,N,s):
    return xc**2/(2*v) - np.log((2*N*xc*(s))/(2*np.pi*v)**0.5)

def constrain_xc_exact(xc,v,s,Ub):
    return np.log(1)-np.log((Ub/(s))*(np.exp(xc**2/(2*v))*np.exp(-(xc-s)**2/(2*v))*((xc+s)/xc) - np.exp(-(s)**2/(2*v)) + np.exp(xc**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(s**2/xc)*(1+erf((s-xc)/np.sqrt(2*v))))) #correct#
    
# if you know XCM, calculate the fold change in Ub...
def calculate_Ub_from_xc(v,sb,xc):

    if xc>sb: # initial guess from MM regime
        logUb_guess = -xc*sb/v+sb**2/2/v + np.log(sb)
    else: # initial guess from QS regime
        logUb_guess = np.log(v/sb**2)-xc**2/2/v + np.log(sb)
        
    condition = lambda y: constrain_xc_exact(xc,v,sb,np.exp(y))
    logUb=fsolve(condition,[logUb_guess])[0]
    return np.exp(logUb)
    
#theoretical predictions for Npfix
def Npfix_small_r(s,r,Ub,N,v):
    q=2*np.log(N*s)/(np.log(s/Ub))
    q_it=2*np.log(N*s)/(np.log(s/Ub))
    xc=fsolve_eq_xc_exact(N,s,Ub,v,q_it*s,1)
    xcm=fsolve_eq_xc_exact(N,r*s,Ub,v,q_it*s,1)
    Npfix=((r*xcm*s)/(xc*s))*np.exp((xc**2-xcm**2)/(2*v))
    return Npfix

def Npfix_large_r(s,r,Ub,N,v):
    q_it=2*np.log(N*s)/(np.log(s/Ub))
    xc=fsolve_eq_xc_exact(N,s,Ub,v,q_it*s,1)
    xcm=fsolve_eq_xc_exact(N,s,Ub,v,xc,r)
    Npfix=2*N*Ub*r*s*xcm/v
    return Npfix

# new version (switches between MM and QS internally, rather than externally)--adjusted for when simulation v unavailable
def calculate_Npfix_twoparam_v_unknown(N,sb,Ub,sm,Um,deltam=0,xc=-1,xcm=-1,correct_xcm=True):
    
    # Standardize how we treat dead end modifiers
    if Um==0 or sm==0:
        # This is a dead end modifier
        # set Um=0 and sm=sb (convention)
        Um=0
        sm=sb
        
    # First make sure we have xc and xcm
    if xc<-0.5:
        # Need to calculate xc from scratch. 
        v,xc = calculate_both_twoparam(N,sb,Ub)
    
    if xcm<-0.5:
        # Need to calculate xcm from scratch
        # maximum possible xcm (from dead-end modifier section)
        xcm_max = np.sqrt(2)*xc*(1-(v/(2*xc**2))*np.log(xc*sb/v*xc**2/v))
        
        if Um>0:
            xcm = calculate_xc_twoparam(v,sm,Um)
            #print("this is xcm ",xcm,xc)
            #print(sm/sb, xcm, xcm_max)
            if correct_xcm and xcm>xcm_max:
            #    #print("Switching xcm!", sm/sb, Um/Ub)
                # Pretend we're a dead end modifier
                xcm=xcm_max
                Um=0
                sm=sb
        else:
        	# We are a dead end modifier
            xcm = xcm_max
    if np.abs(xcm-xc)<.00001:
        xcm=xc
    
    # Next gate on whether we are in the MM, QS, (or DE) regimes 
    if Um==0:
        # use dead end solution 
        
        if xc+deltam>xcm:
            Npfix=(v/(xc*sb))*(np.exp((xc**2-(xcm-deltam)**2)/(2*v))-1) +2*N*deltam*(gaussian_cdf((deltam-xcm)/np.sqrt(v))-gaussian_cdf(-xc/np.sqrt(v)))
        else:
            Npfix=0 
        
    elif xcm>sm:
        # use MM solution
        
        base_Npfix = (xcm/xc)*(sm/sb)*np.exp((xc**2-xcm**2)/(2*v))
        
        # Next gate on direct cost or benefit
        if deltam == 0: # No cost or benefit
            Npfix = base_Npfix
        
        elif deltam<0: # Direct cost
        
            # define positive version of cost (for convenience)
            abs_deltam = np.abs(deltam)
            # switch variables to k and Delta
            k=np.floor(abs_deltam/sm)
            D=abs_deltam-k*sm
            
            Npfix = base_Npfix*np.exp(-(xcm-sm/2)*abs_deltam/v+D*(sm-D)/(2*v)-gammaln(k+1)+k*np.log(1-sm/xcm))*((1-np.exp(-D*np.minimum(sm,sm-D+xc-xcm)/v))/(sm*D/v)*(1-np.exp(-sm*(sm-D)/v))+((xcm-sm)/xcm)/(k+1)*((1-np.exp(-sm*(sm-D)/v))/(sm*(sm-D)/v))*(1-np.exp(-D*sm/v)))
        else: # Direct benefit
            
            if xc+deltam > xcm:
                Npfix = base_Npfix*(np.exp(xcm*deltam/v-deltam**2/(2*v))*(1-np.exp(-sm*deltam/v))/(sm*deltam/v))+2*N*deltam*gaussian_cdf((deltam-xcm)/np.sqrt(v))
                
            else:
                # switch variables to k and Delta
                k=np.floor((xcm-xc-deltam)/sm)
                
                D=(xcm-xc-deltam)-k*sm
                Npfix = base_Npfix * np.exp( xcm*deltam/v-deltam**2/2/v+k*np.log(1-sm/xcm)-k*(k-1)*sm**2/2/v-(k*sm+D)*deltam/v-k*sm*D/v-gammaln(k+1))*(1-np.exp(-(deltam+k*sm)*(sm-D)/v))/((deltam+k*sm)*sm/v)
                
                #if k>-0.5:
                #   print("Using k>=0!", k, sm/sb,deltam/sb,Npfix)
                
            
    else:
        # use QS solution
        base_Npfix=2*N*Um*sm*xcm/v # most simplified version
        #Npfix=((xcm*xcm)/(xc*sb))*np.exp((xc**2-xcm**2)/(2*v)) # less simplified version
        # is continuous, but not asymptotically correct at large sm/s
    
        # Next gate on direct cost or benefit
        if deltam == 0:
            Npfix = base_Npfix
        elif deltam < 0:
            # define positive version of cost (for convenience)
            abs_deltam = np.abs(deltam)
            # switch variables to k and Delta
            k=np.floor(abs_deltam/sm)
            D=abs_deltam-k*sm
        
            # We do numerical integration of the expression in the SI
            first_term = lambda x: (Um/sm)**k*(1/factorial(k))*np.exp(-k*sm*D/v-D**2/(2*v)-x*D/v)*0.5*(erf((x+k*sm)/np.sqrt(2*v))+1)*(1/xcm)
            
            second_term = lambda x:  (Um/sm)**k*(v/(sm*xcm*np.sqrt(2*np.pi*v)))*np.exp(-(x-deltam)**2/(2*v))*((x+(k+1)*sm)/sm)*gamma(-x/sm-k)/gamma(1-x/sm) 
            
            third_term = lambda x: (v/(sm*np.sqrt(2*np.pi*v)))*np.exp(-(x-deltam)**2/(2*v))*(1/factorial(k))*(sm/(x+k*sm))*(1/xcm)*(Ub/sm)**k
            
            first_integrand = lambda x: first_term(x)+second_term(x)+third_term(x)
            
            second_integrand = lambda x: (Um/sm)**(k+1)*(np.exp(-k*sm*D/v-D**2/(2*v)+(2*k+1)*sm**2/(2*v)))*(1/factorial(k+1))*np.exp(x*(sm-D)/v)*0.5*(erf((x+(k+1)*sm)/np.sqrt(2*v))+1)*(1/xcm)
            
            first_integral = integrate.quad(first_integrand,xcm-(k+1)*sm,np.minimum(xcm-k*sm,xc-k*sm-D))[0]
            
            second_integral = integrate.quad(second_integrand,xcm-(k+2)*sm,xcm-(k+1)*sm)[0]
            
            Npfix = base_Npfix*(first_integral+second_integral)
            
        else:
            print("Not suported yet!")
            Npfix = -1
            
    return Npfix

# new version (switches between MM and QS internally, rather than externally)
def calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um,deltam=0,xc=-1,xcm=-1,correct_xcm=True):
    
    # Standardize how we treat dead end modifiers
    if Um==0 or sm==0:
        # This is a dead end modifier
        # set Um=0 and sm=sb (convention)
        Um=0
        sm=sb
        
    # First make sure we have xc and xcm
    if xc<-0.5:
        # Need to calculate xc from scratch. 
        xc = calculate_xc_twoparam(v,sb,Ub)
    
    if xcm<-0.5:
        # Need to calculate xcm from scratch
        # maximum possible xcm (from dead-end modifier section)
        xcm_max = np.sqrt(2)*xc*(1-(v/(2*xc**2))*np.log(xc*sb/v*xc**2/v))
        
        if Um>0:
            xcm = calculate_xc_twoparam(v,sm,Um)
            #print(sm/sb, xcm, xcm_max)
            if correct_xcm and xcm>xcm_max:
                #print("Switching xcm!", sm/sb, Um/Ub)
                # Pretend we're a dead end modifier
                xcm=xcm_max
                Um=0
                sm=sb
        else:
        	# We are a dead end modifier
            xcm = xcm_max
    
    # Next gate on whether we are in the MM, QS, (or DE) regimes 
    if Um==0:
        # use dead end solution 
        
        if xc+deltam>xcm:
            Npfix=(v/(xc*sb))*(np.exp((xc**2-(xcm-deltam)**2)/(2*v))-1) +2*N*deltam*(gaussian_cdf((deltam-xcm)/np.sqrt(v))-gaussian_cdf(-xc/np.sqrt(v)))
        else:
            Npfix=0 
        
    elif xcm>sm:
        # use MM solution
        
        base_Npfix = (xcm/xc)*(sm/sb)*np.exp((xc**2-xcm**2)/(2*v))
        
        # Next gate on direct cost or benefit
        if deltam == 0: # No cost or benefit
            Npfix = base_Npfix
        
        elif deltam<0: # Direct cost
        
            # define positive version of cost (for convenience)
            abs_deltam = np.abs(deltam)
            # switch variables to k and Delta
            k=np.floor(abs_deltam/sm)
            D=abs_deltam-k*sm
            
            Npfix = base_Npfix*np.exp(-(xcm-sm/2)*abs_deltam/v+D*(sm-D)/(2*v)-gammaln(k+1)+k*np.log(1-sm/xcm))*((1-np.exp(-D*np.minimum(sm,sm-D+xc-xcm)/v))/(sm*D/v)*(1-np.exp(-sm*(sm-D)/v))+((xcm-sm)/xcm)/(k+1)*((1-np.exp(-sm*(sm-D)/v))/(sm*(sm-D)/v))*(1-np.exp(-D*sm/v)))
        
        else: # Direct benefit
            
            if xc+deltam > xcm:
                Npfix = base_Npfix*(np.exp(xcm*deltam/v-deltam**2/(2*v))*(1-np.exp(-sm*deltam/v))/(sm*deltam/v))+2*N*deltam*gaussian_cdf((deltam-xcm)/np.sqrt(v))
                
            else:
                # switch variables to k and Delta
                k=np.floor((xcm-xc-deltam)/sm)
                
                D=(xcm-xc-deltam)-k*sm
                Npfix = base_Npfix * np.exp( xcm*deltam/v-deltam**2/2/v+k*np.log(1-sm/xcm)-k*(k-1)*sm**2/2/v-(k*sm+D)*deltam/v-k*sm*D/v-gammaln(k+1))*(1-np.exp(-(deltam+k*sm)*(sm-D)/v))/((deltam+k*sm)*sm/v)
                
                #if k>-0.5:
                #   print("Using k>=0!", k, sm/sb,deltam/sb,Npfix)
                
            
    else:
        # use QS solution
        base_Npfix=2*N*Um*sm*xcm/v # most simplified version
        #Npfix=((xcm*xcm)/(xc*sb))*np.exp((xc**2-xcm**2)/(2*v)) # less simplified version
        # is continuous, but not asymptotically correct at large sm/s
    
        # Next gate on direct cost or benefit
        if deltam == 0:
            Npfix = base_Npfix
        elif deltam < 0:
            # define positive version of cost (for convenience)
            abs_deltam = np.abs(deltam)
            # switch variables to k and Delta
            k=np.floor(abs_deltam/sm)
            D=abs_deltam-k*sm
        
            # We do numerical integration of the expression in the SI
            first_term = lambda x: (Um/sm)**k*(1/factorial(k))*np.exp(-k*sm*D/v-D**2/(2*v)-x*D/v)*0.5*(erf((x+k*sm)/np.sqrt(2*v))+1)*(1/xcm)
            
            second_term = lambda x:  (Um/sm)**k*(v/(sm*xcm*np.sqrt(2*np.pi*v)))*np.exp(-(x-deltam)**2/(2*v))*((x+(k+1)*sm)/sm)*gamma(-x/sm-k)/gamma(1-x/sm) 
            
            third_term = lambda x: (v/(sm*np.sqrt(2*np.pi*v)))*np.exp(-(x-deltam)**2/(2*v))*(1/factorial(k))*(sm/(x+k*sm))*(1/xcm)*(Ub/sm)**k
            
            first_integrand = lambda x: first_term(x)+second_term(x)+third_term(x)
            
            second_integrand = lambda x: (Um/sm)**(k+1)*(np.exp(-k*sm*D/v-D**2/(2*v)+(2*k+1)*sm**2/(2*v)))*(1/factorial(k+1))*np.exp(x*(sm-D)/v)*0.5*(erf((x+(k+1)*sm)/np.sqrt(2*v))+1)*(1/xcm)
            
            first_integral = integrate.quad(first_integrand,xcm-(k+1)*sm,np.minimum(xcm-k*sm,xc-k*sm-D))[0]
            
            second_integral = integrate.quad(second_integrand,xcm-(k+2)*sm,xcm-(k+1)*sm)[0]
            
            Npfix = base_Npfix*(first_integral+second_integral)
            
        else:
            print("Not suported yet!")
            Npfix = -1
            
    return Npfix

# Calculates the critical cost or benefit necessary to make modifier neutral
def calculate_critical_cost_twoparam(N,sb,Ub,v,sm,Um):

    # First try to calculate a guess.
    
    # Calculate base Npfix:
    base_Npfix = calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um)
    
    xc = calculate_xc_twoparam(v,sb,Ub)
    
    xcm_max = np.sqrt(2)*xc*(1-(v/(2*xc**2))*np.log(xc*sb/v*xc**2/v))
    
    if Um>0:
    	xcm = calculate_xc_twoparam(v,sm,Um)
    else:
    	xcm=xcm_max
    
    if xcm>xcm_max:
        print("Switching xcm!", sm/sb, Um/Ub)
        xcm=xcm_max
        #Um=0
        #sm=sb
    
    
    print("base npfix", base_Npfix)
    if base_Npfix < 1: # It's a costly modifier, so looking for a fitness benefit
        if base_Npfix==0:
            # It's a dead end modifier!
            #deltam_guess=xcm-xc+v/xc*np.log(xc*sb/v)
            deltam_guess = (np.sqrt(2)-1)*xc
        else:
            # It's not
            deltam_guess = -v/xcm*np.log(base_Npfix)
    elif base_Npfix>1: # It's a beneficial modifier, so looking for a fitness cost
            deltam_guess = -v/xcm*np.log(base_Npfix)
    else: # It's exactly neutral
        deltam_guess = 0
        
    # Then calculate it numerically
    calculate_logNpfix = lambda x: np.log(calculate_Npfix_twoparam(N,sb,Ub,v,sm,Um,deltam=x)+1e-09)
    
    deltam_critical = fsolve(calculate_logNpfix,deltam_guess)[0]
    
    #print("Rootfinding result:", deltam_guess, deltam_critical, sm/sb)
    #print("Err:", deltam_guess, calculate_logNpfix(deltam_critical))
    
    return deltam_critical

#Npfix for joint sb and Ub modifier
def Npfix_supply_effect_alt(s,rUs,Ub,N,rs,v):
    out=[]
    for rU in rUs:
        r=rs
        q=2*np.log(N*s)/(np.log(s/Ub))
        xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
        xcm=fsolve_eq_xc_exact(N,r*s,rU*Ub,v,q*s,1) 
        Npfix=(xcm*r/(xc))*np.exp((xc**2-xcm**2)/(2*v))
        out.append(Npfix)
    return np.asarray(out)

def Npfix_mutation_rate(s,rs,Ub,N,v):
    out=[]
    for r in rs:
        q_it=2*np.log(N*s)/(np.log(s/Ub))
        xc=fsolve_eq_xc_exact(N,s,Ub,v,q_it*s,1)
        xcm=fsolve_eq_xc_exact(N,s,r*Ub,v,q_it*s,1)
        
        Npfix1=(xcm*s)/(xc*s)*np.exp((xc**2-xcm**2)/(2*v))
        Npfix2=2*N*r*Ub*s*xcm/v
        Npfix=np.minimum(Npfix1,Npfix2)
        out.append(Npfix)
    return np.asarray(out)

def Npfix_mutation_rate_ind(s,rs,Ub,N,v):
    out=[]
    q_it=2*np.log(N*s)/(np.log(s/Ub))
    xc=fsolve_eq_xc_exact(N,s,Ub,v,q_it*s,1)
    xcm=xc-(v/s)*np.log(rs)
    q=xc/s
    Npfix=(np.exp(q*np.log(r)-(v/(2*s**2))*np.log(r)**2)) + (v/(xc*s))*((np.exp(q*np.log(r)-(v/(2*s**2))*np.log(r)**2))-1)
    out.append(Npfix)
    return np.asarray(out)

#Npfix for joint sb and Ub modifier
def Npfix_supply_effect_alt(s,rUs,Ub,N,rs,v):
    out=[]
    for rU in rUs:
        r=rs
        q=2*np.log(N*s)/(np.log(s/Ub))
        xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
        xcm=fsolve_eq_xc_exact(N,r*s,rU*Ub,v,q*s,1) 
        Npfix=(xcm*r/(xc))*np.exp((xc**2-xcm**2)/(2*v))
        out.append(Npfix)
    return np.asarray(out)

#Additive Npfix predictions=product of individual Npfix's
def Npfix_supply_effect_wrong_alt(s,rUs,Ub,N,rs,v):
    out=[]
    for rU in rUs:
        r=rs
        q=2*np.log(N*s)/(np.log(s/Ub))
        xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
        xcm_rs=fsolve_eq_xc_exact(N,r*s,Ub,v,q*s,1)
        xcm_rU=fsolve_eq_xc_exact(N,s,rU*Ub,v,q*s,1)
        Npfix_rs=(xcm_rs*r*s/(xc*s))*np.exp((xc**2-xcm_rs**2)/(2*v))
        Npfix_rU=(xcm_rU*s/(xc*s))*np.exp((xc**2-xcm_rU**2)/(2*v))
        Npfix=Npfix_rs*Npfix_rU
        out.append(Npfix)
    return np.asarray(out)

########
#
# Direct costs and benefits
#
########   

#def calculate_Npfix_direct_cost(xc)
#def calculate_Npfix_twoparam_cost(N,sb,Ub,v,sm,Um,deltam,xc=-1,xcm=-1):
#calculate_logNpfix_beneficial(s,xc,v,sb,Ub):   

#Previous Theoretical predction for direct costs
def Npfix_del(s,r,Ub,N,sds,v):
    out=[]
    q=(2*np.log(N*s)/(np.log(s/Ub)))
    xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
    xcm=fsolve_eq_xc_exact(N,s,Ub,v,xc,r)
    Npfix_nm=2*N*Ub*xcm*r*s/v
    sm=r*s
    for sd in sds:
        k=np.floor(np.abs(sd)/(sm))
        D=np.abs(sd)-k*r*s
        if r<np.sqrt(2*xc/s):
            sd=-sd
            Npfix=(xcm*sm/(xc*s))*np.exp((xc**2-xcm**2)/(2*v))*np.exp(-(xcm-sm/2)*np.abs(sd)/v)*np.exp(-D**2/(2*v)+sm*D/(2*v))/factorial(k)*((1-sm/xcm)**k)*((1-np.exp(-D*np.minimum(sm,sm-D+xc-xcm)/v))/(sm*D/v)*(1-np.exp(-sm*(sm-D)/v))+((xcm-sm)/xcm)/(k+1)*((1-np.exp(-sm*(sm-D)/v))/((sm*(sm-D)/v))*(1-np.exp(-D*sm/v))))
            out.append(Npfix)
        else:
            #numerical integration
            def integrand(x,r,s,sd,v,k):
                return (Ub/sm)**k*(v/(sm*xcm*np.sqrt(2*np.pi*v)))*np.exp(-(x-sd)**2/(2*v))*((x+(k+1)*sm)/sm)*gamma(-x/sm-k)/gamma(1-x/sm) +(Ub/sm)**k*(1/factorial(k))*np.exp(-k*sm*D/v-D**2/(2*v)-x*D/v)*0.5*(erf((x+k*sm)/np.sqrt(2*v))+1)*(1/xcm)+(v/(sm*np.sqrt(2*np.pi*v)))*np.exp(-(x-sd)**2/(2*v))*(1/factorial(k))*(sm/(x+k*sm))*(1/xcm)*(Ub/sm)**k
            def integrand_extra(x,r,s,sd,v,k):
                return (Ub/sm)**(k+1)*(np.exp(-k*sm*D/v-D**2/(2*v)+(2*k+1)*sm**2/(2*v)))*(1/factorial(k+1))*np.exp(x*(sm-D)/v)*0.5*(erf((x+(k+1)*sm)/np.sqrt(2*v))+1)*(1/xcm)
            val=integrate.quad(integrand,xcm-(k+1)*sm,np.minimum(xcm-k*sm,xc-k*sm-D),args=(r,s,sd,v,k))
            val_e=integrate.quad(integrand_extra,xcm-(k+2)*sm,xcm-(k+1)*sm,args=(r,s,sd,v,k))
            Npfix=Npfix_nm*(val[0]+val_e[0])
            out.append(Npfix)
    return np.asarray(out)
    
# Previous theoretical prediction for deleterious modifier with direct benefit   
def Npfix_ben(s,r,Ub,N,sds,v):
    out=[]
    q=(2*np.log(N*s)/(np.log(s/Ub)))
    xc=fsolve_eq_xc_exact(N,s,Ub,v,q*s,1)
    q=xc/s
    sm=r*s
    if r>0:
        xcm1=1/r*(xc+s/2*(r**2-1)+(v/s)*np.log(r))
        xcm=fsolve_eq_xc_exact(N,r*s,Ub,v,xcm1,1)
        e=v/xc*np.log(xc*s/v*xc**2/v)
        xcm_z=np.sqrt(2)*xc*(1-(v/(2*xc**2))*np.log(xc*s/v*xc**2/v))
        xcm=np.minimum(xcm,xcm_z)
    for sd in sds:
        #evolutionary dead-end
        if r==0:
            
            e=v/xc*np.log(xc*s/v*xc**2/v)
            xcm=np.sqrt(2)*xc*(1-(v/(2*xc**2))*np.log(xc*s/v*xc**2/v))
            if xc-sd>xcm:
                Npfix=(v/(xc*s))*(np.exp((xc**2-(xcm+sd)**2)/(2*v))-1) -N*(sd)*(erf((-xcm-sd)/(2*v)**0.5)-erf(-xc/np.sqrt(2*v)))
                out.append(Npfix)
            else:
                Npfix=0
                out.append(Npfix)
        #deleterious modifier with direct benefit
        elif 0<r and r*s<np.sqrt(2*xc*s):
            if xc-sd>xcm:
                sd=-sd
                Npfix=(xcm*sm/(xc*s))*np.exp((xc**2-xcm**2)/(2*v))*(np.exp(xcm*sd/v-sd**2/(2*v))*(1-np.exp(-sm*sd/v))/((sm*sd/v)))+N*sd*(1+erf((sd-xcm)/np.sqrt(2*v)))
            else:
                Npfix=-((v*xcm/(s*sd*xc)))*np.exp((xc**2-xcm**2-sd**2)/(2*v))*(np.exp(-sd*(-sd+xc)/v)-np.exp(-(xcm-r*s)*sd/v))
            out.append(Npfix)
    return np.asarray(out)

    
########
#
# Continuous DFEs
#
########

# TO DO NEXT:

# NEED THING TO SOLVE FOR SBEFF,UBEFF given xc
# DO IT FOR EXPONENTIAL AND OTHER BETA (COMBINE)
# DO IT FOR BETA PLUS DELTA (USES BETA ALONE), then takes argmax
# USE TWO-PARAM XCM DEFINITION composed with the sbmeff as a lambda.
# NOT CLEAR WHICH ORDER TO DO THE QS THING....

# previous version
def fsolve_eq_xc_general(s,Ub,v,xc_pop,r,b):
    guess_xc=xc_pop/r
    root=fsolve(constrain_xc_general,[guess_xc],args=(v,s,Ub,b))[0]
    return root

def constrain_xc_general(xc,v,s,Ub,b):
    if b==10:
        s_star_guess=((xc*(s**b)/(b*v))**(1/(b-1)))*(1-1/(1+b*(b-1)*(v/s**2)*(xc/(b*v))**((b-2)/(b-1))))
        delta_guess=((1/v)+((b*(b-1))/(s**2))*(s_star_guess/s)**(b-2))**(-0.5)
        s_star,delta=fsolve(equations_both_s_star_delta,[s_star_guess,delta_guess],args=(xc,v,s,b))
        Ub=Ub*short_tailed_exponential(s_star,s,10)*(2*np.pi*delta**2)**0.5
        s=s_star
        return np.log((Ub/(s))*(np.exp(xc**2/(2*v))*np.exp(-(xc-s)**2/(2*v)) - np.exp(-(s)**2/(2*v)) + np.exp(xc**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(s**2/xc)*(1+erf((s-xc)/np.sqrt(2*v)))))
        #return np.log(((s_star/Ub)*(1/((2*np.pi*delta**2)**(0.5)*short_tailed_exponential(s_star,s,b)))*(1-s_star/xc)))-xc*s_star/v+(s_star)**2/(2*v)
    elif b==1:
        s_star=xc-v/s #beta=1
        Ub=Ub*short_tailed_exponential(s_star,s,1)*(2*np.pi*v)**0.5
        s=s_star
        return np.log((Ub/(s))*(np.exp(xc**2/(2*v))*np.exp(-(xc-s)**2/(2*v)) - np.exp(-(s)**2/(2*v)) + np.exp(xc**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(s**2/xc)*(1+erf((s-xc)/np.sqrt(2*v)))))
        #return np.log((Ub/s)*np.sqrt(2*np.pi*v)*(1/(xc-v/s)+s/v*(1+s/xc))*np.exp((xc-v/s)**2/(2*v)))

def s_star_beta(s_star,xc,v,s,b):
    return s_star-xc+v*((b/s)*((s_star/s)**(b)))

def delta_beta(s_star,delta,xc,v,s,b):
    return delta-((1/v)+((b-1)*b*((s_star/s)**(b))/(s_star**2)))**(-1/2)

def equations_both_s_star_delta(p,xc,v,s,b):
    s_star,delta=p
    return [s_star_beta(s_star,xc,v,s,b),delta_beta(s_star,delta,xc,v,s,b)]

def fsolve_eq_xc_general_large_r(s,Ub,v,xc_pop,r,b):
    guess_xc=.02
    root=fsolve(constrain_xc_general_large_r,[guess_xc],args=(v,s,Ub,b))[0]
    return root

def constrain_xc_general_large_r(xc,v,s,Ub,b):
    if b==10:

        s_star_guess=((xc*(s**b)/(b*v))**(1/(b-1)))*(1-1/(1+b*(b-1)*(v/s**2)*(xc/(b*v))**((b-2)/(b-1))))
        delta_guess=((1/v)+((b*(b-1))/(s**2))*(s_star_guess/s)**(b-2))**(-0.5)
        s_star,delta=fsolve(equations_both_s_star_delta,[s_star_guess,delta_guess],args=(xc,v,s,b))
        return np.log(((Ub*np.sqrt(2*np.pi*delta**2)*short_tailed_exponential(s_star,s,b))/(s_star))*np.exp(-s_star**2/(2*v)+xc*s_star/v)+0.5*np.exp(xc**2/(2*v))*((s_star*2*np.pi*delta*short_tailed_exponential(s_star,s,b)*Ub)/(np.sqrt(v)*xc)))

def short_tailed_exponential(x,s,b):
    return (1/s)*(1/(gamma(1+1/b)))*np.exp(-(x/s)**b)

#solve for xc numerically
def fsolve_eq_xc_exact_p(N,s,Ub,sm,Ubm,v,xc_pop,r):
    xc=2*s*np.log(N*s)/np.log(s/Ub)
    if r<np.sqrt(2*xc*s):
        guess_xc=(1/r)*(xc_pop+(s/2)*(r**2-1))
    else:
        guess_xc=np.sqrt(2*v)*np.log(v/(Ub*r*s))**(0.5)
    root=fsolve(constrain_xc_exact_p,[guess_xc],args=(v,r*s,Ub,sm,Ubm,N))[0]
    return root

def fsolve_eq_xc_exact_p_alt(N,s,Ub,sm,Ubm,v,xc_pop,r,xc):
    xc_g=2*s*np.log(N*s)/np.log(s/Ub)
    if r<np.sqrt(2*xc*s):
        guess_xc=(1/r)*(xc_pop+(s/2)*(r**2-1))
    else:
        guess_xc=np.sqrt(2*v)*np.log(v/(Ub*r*s))**(0.5)
    root=fsolve(constrain_xc_exact_p_alt,[guess_xc],args=(xc,v,r*s,Ub,sm,Ubm,N))[0]
    return root

def constrain_xc_exact_p(xc,v,s,Ub,sm,Ubm,N):
    a=(Ub/(s))*(np.exp(xc**2/(2*v))*np.exp(-(xc-s)**2/(2*v)) - np.exp(-(s)**2/(2*v)) + np.exp(xc**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(s**2/xc)*(1+erf((s-xc)/np.sqrt(2*v)))) #correct#
    b=(Ubm/(sm))*(np.exp(xc**2/(2*v))*np.exp(-(xc-sm)**2/(2*v)) - np.exp(-(sm)**2/(2*v)) + np.exp(xc**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(sm**2/xc)*(1+erf((sm-xc)/np.sqrt(2*v)))) #correct#
    return np.log(a+b)

def constrain_xc_exact_p_alt(xcm,xc,v,s,Ub,sm,Ubm,N):
    a=np.exp(xcm*s/v-s**2/(2*v))
    b=(Ubm/Ub)*(v/s)*(np.exp(xcm**2/(2*v))*np.exp(-(xcm-sm)**2/(2*v)) - np.exp(-(sm)**2/(2*v)) + np.exp(xcm**2/(2*v))*(np.sqrt(np.pi/(2*v)))*(sm**2/xcm)*(1+erf((sm-xcm)/np.sqrt(2*v)))) #correct#
    print(a,b,xc,xcm)
    return np.log(b/a)-np.log(xc-xcm)

def calculate_seff_Ueff_beta(v,xc,s0,U0,b):

    # Working out the average
    #y = x^beta --> x = y^(1/beta)
    #dy = beta*x^(beta-1) dx = beta*y^((beta-1)/beta) dx
    #dx = 1/beta*y^(-1+1/beta)
    
    #norm = integral 1/beta/Gamma(1+1/beta) * y^(1/beta-1)*exp(-y) = 1/beta Gamma(1/beta) / Gamma(1+1/beta) = 1
    
    #savg = integral 1/beta*1/Gamma(1+1/\beta) * y^(2/beta-2) * exp(-y) = Gamma(2/beta)/beta Gamma(1+1/beta) = Gamma(2/beta)/Gamma(1/beta) 
    
    savg = s0*gamma(2.0/b)/gamma(1.0/b)
    
    if savg > xc:
        return savg, U0
    
    if b==1:
        sstar = xc-v/s0
    else:    
        sstar_condition = lambda x: x-xc+v*(b/s0)*np.power((x/s0),b-1)
    
        sstar_guess = ((xc*(s0**b)/(b*v))**(1/(b-1)))*(1-1/(1+b*(b-1)*(v/s0**2)*(xc/(b*v))**((b-2)/(b-1))))

        sstar = fsolve(sstar_condition, [sstar_guess])[0]
    
    delta = ((1/v)+((b-1)*b*((sstar/s0)**(b))/(sstar**2)))**(-1/2)
    Ustar = U0*np.sqrt(2*np.pi)*delta/s0/gamma(1+1.0/b)*np.exp(-np.power(sstar/s0,b))
    
    #print(delta,np.sqrt(v)) # these are the same
    
    if sstar < xc:
        # Still in MM regime, return sstar, Ustar
        return sstar, Ustar
    else:
        # actually in QS regime, return savg, U0
        # print("Shouldn't be here yet", xc,sstar,savg)
        return savg, U0
        
def calculate_xc_seff_Ueff_beta(v,s0,U0,b):
    
    # Construct initial guess of xc
    if b==1:
        guess_xc = np.sqrt(2*v*np.log(s0/U0))
    else:
        guess_xc = v/s0*np.log(np.sqrt(b)*s0/U0)
    
    # Helper function that takes the output of calculate_seff_Ueff
    constrain_xc_tuple = lambda x,y: constrain_xc_exact(x,v,y[0],y[1])
    # xc constraint using the corresponding values of seff,Ueff
    constrain_xc_beta = lambda x: constrain_xc_tuple(x,calculate_seff_Ueff_beta(v,x,s0,U0,b)) 
    
    # Solve for xc
    xc=fsolve(constrain_xc_beta,guess_xc)[0]
    # Calculate the associated values of seff,Ueff
    seff,Ueff = calculate_seff_Ueff_beta(v,xc,s0,U0,b)
    
    return xc,seff,Ueff

def calculate_seff_Ueff_beta_plus_delta(v,xc,s0,U0,b,s1,U1):
    
    sstar,Ustar = calculate_seff_Ueff_beta(v,xc,s0,U0,b)
    
    log_W0 = calculate_logNpfix_beneficial(sstar,xc,v,sstar,Ustar)+np.log(Ustar)
    log_W1 = calculate_logNpfix_beneficial(s1,xc,v,sstar,Ustar)+np.log(U1)
    
    if log_W1 > log_W0:
        #print("Returning dmu!")
        return s1,U1
    else:
        #print("Returning original!")
        return sstar,Ustar

def calculate_xc_seff_Ueff_beta_plus_delta(v,s0,U0,b,s1,U1):
    
    # Construct initial guess of xc
    if b==1:
        guess_xc = np.sqrt(2*v*np.log(s0/U0))
    else:
        guess_xc = v/s0*np.log(np.sqrt(b)*s0/U0)
    
    seff,Ueff = calculate_seff_Ueff_beta_plus_delta(v,guess_xc,s0,U0,b,s1,U1)
    
    new_guess_xc = calculate_xc_twoparam(v,seff,Ueff)
    #print("Guess:", guess_xc, new_guess_xc, seff, Ueff)
    guess_xc = new_guess_xc
    
    # Helper function that takes the output of calculate_seff_Ueff
    constrain_xc_tuple = lambda x,y: constrain_xc_exact(x,v,y[0],y[1])
    # xc constraint using the corresponding values of seff,Ueff
    constrain_xc_beta_plus_delta = lambda x: constrain_xc_tuple(x,calculate_seff_Ueff_beta_plus_delta(v,x,s0,U0,b,s1,U1)) 
    
    # Solve for xc
    xc=fsolve(constrain_xc_beta_plus_delta,guess_xc)[0]
    # Calculate the associated values of seff,Ueff
    seff,Ueff = calculate_seff_Ueff_beta_plus_delta(v,xc,s0,U0,b,s1,U1)
       
    return xc,seff,Ueff

# Calculate's dxc in the perturbative regime    
def calculate_I_delta(xc,v,sb,Ub,s1,U1):

    # Define a unitless version of dxc
    # y = dxc*sb/v 
    
    # Define condition that dxc must satisfy (from perturbative regime in paper)
    # (tried expm1(y) instead of y.. doens't work as well...)
    # dxc_condition = lambda y: y+np.exp(y*s1/sb+xc*s1/v-s1**2/2/v - y - xc*sb/v + sb**2/2/v - np.log(Ub/sb) + np.log(U1/s1))
    dxc_condition = lambda y: y+np.exp(calculate_logNpfix_beneficial(s1,np.fmax(1e-09,xc+y*v/sb),v,sb,Ub)-calculate_logNpfix_beneficial(sb,np.fmax(1e-09,xc+y*v/sb),v,sb,Ub)+np.log(U1)-np.log(Ub))


    # The lowest deltaxc can be is -xc
    # so check to see if a solution is even possible
    ymin = -xc*sb/v
     
    #if dxc_condition(ymin) > 0:
    #    return -1*ymin
    
    # Recalculate using slightly more sensitive version...
    # (doesn't do anything...)
    
    guess_y = -1*np.exp(calculate_logNpfix_beneficial(s1,xc,v,sb,Ub)-calculate_logNpfix_beneficial(sb,xc,v,sb,Ub)+np.log(U1)-np.log(Ub))
    
    # if solution is possible, find it numerically
    y = fsolve(dxc_condition, guess_y)[0]
    
    if y<ymin:
        I = -1*ymin
    elif guess_y < ymin:
        I=-1*ymin
    elif dxc_condition(y)>1e-02:
        # too much error!
        I = 1e09
    else:
        I = -1*y
        
    return I
    
# new version (switches between MM and QS internally, rather than externally)
def calculate_Npfix_beta_plus_delta(N,s0,U0,beta,s1,U1,v,deltam=0,perturbative_transition=1.0):
    
    # Need to calculate xc from scratch. 
    xc,seff,Ueff = calculate_xc_seff_Ueff_beta(v,s0,U0,beta)
    
    I = calculate_I_delta(xc,v,seff,Ueff,s1,U1)
    #print("I =", I)
    
    perturbative_xcm = xc-np.clip(I,-perturbative_transition,perturbative_transition)*v/seff
    perturbative_smeff = seff
    perturbative_Umeff = calculate_Ub_from_xc(v,perturbative_smeff,perturbative_xcm)
    
    if I<perturbative_transition:
        # We are in the perturbative regime
        xcm=perturbative_xcm
        smeff=perturbative_smeff
        Umeff=perturbative_Umeff
    else:
        # We are in the modifier dominated regime (or the crossover region)
        
        # Need to recalculate xcm, smeff, Umeff:
        xcm,smeff,Umeff = calculate_xc_seff_Ueff_beta_plus_delta(v,s0,U0,beta,s1,U1)
        
        # Check if we're in the crossover region
        # (if we are, pretend we are in the perturbative regime...)
        base_Npfix = calculate_Npfix_twoparam(N,seff,Ueff,v,smeff,Umeff,xc=xc,xcm=xcm,deltam=0)
        
        perturbative_base_Npfix = calculate_Npfix_twoparam(N, seff, Ueff, v, perturbative_smeff, perturbative_Umeff, xc=xc, xcm=perturbative_xcm, deltam=0)
        
        if np.fabs(np.log(base_Npfix)) < np.log(perturbative_base_Npfix):
            # secretly in the perturbative regime!
            xcm = perturbative_xcm
            smeff = perturbative_smeff
            Umeff = perturbative_Umeff
    
    #print("Calculating for:",seff,Ueff,smeff,Umeff,deltam,xc,xcm)        
    Npfix = calculate_Npfix_twoparam(N,seff,Ueff,v,smeff,Umeff,xc=xc,xcm=xcm,deltam=deltam) 
    
    return Npfix

# Switching from one beta distribution to another...    
def calculate_Npfix_beta(N,s0,U0,beta0,s1,U1,beta1,v,deltam=0):
    
    # Need to calculate xc from scratch. 
    xc,seff,Ueff = calculate_xc_seff_Ueff_beta(v,s0,U0,beta0)
    
    # Need to calculate xcm from scratch.
    if U1==0: # dead end
    	xcm=0
    	smeff=seff
    	Umeff=0
    else:
    	xcm,smeff,Umeff = calculate_xc_seff_Ueff_beta(v,s1,U1,beta1)
    
    Npfix = calculate_Npfix_twoparam(N,seff,Ueff,v,smeff,Umeff,deltam=deltam) 
    
    return Npfix


    
    
