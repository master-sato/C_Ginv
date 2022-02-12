'''
Example of Continuation/Generalized-inverse (G/Ginv) method
(Two-link arm)
Made in Feb. 2022 ver.0.1
        Feb. 2022 ver. 0.2
            Splitting time step (dt) of system time evolution and
            sampling period (SamplingT). The input value is on hold
            till the next sampling peirod, making a discrete input
            to a continuous state equation.
        Feb. 2022 ver. 0.2.1
            Bug fixed.


BSD 2-Clause License

Copyright (c) 2022, Susumu Morita
All rights reserved.



Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''


import numpy as np
import matplotlib.pyplot as plt
from CGinv import C_Ginv
import time








###########################
## simulation parameters ##
###########################
state_dim=4       # state dimension
input_dim=2       # input dimension
diff_order=2      # k of u^(k)=0

t0=0.0            # initial time [s]
T=0.2             # Horizon [s]
N=2               # Integration steps within the MPC computation

SamplingT=0.0125     # Sampling period [s]
dt=0.001           # Time step for time evolution [s]
Tf=5              # Simulation duration [s]
max_iter=int((Tf-t0)/dt)+1   # iteration of simulation (for loop iteration)
zeta = 1/SamplingT       # damping parameter for C/Ginv

delta=SamplingT/20    # Window width to catch sampling period timing
                      # Try (Sampling period)/10 to (Sampling period)/30


## parameters for Gauss-Newton methods ##
tol = 1e-5           # terminates iteration when norm(Func) < tol
max_iterations = 15  # maximum iteration of Gauss-Newton method
k = 1                # damping coefficient inside Gauss-Newton method


## file_name for saving graphs ##
file_name='CGinv_Two-linkArm_T'+str(T)+'N'+str(N)+'dt'+str(dt)+'DiffOrd'+str(diff_order)


####################
##  Initial state ##
####################
x_init=np.zeros(state_dim)
x_init[0]=-np.pi/180*45
x_init[1]=-np.pi/180*60


###################
##  target state ##
###################
x_ref=np.zeros(state_dim)





## vector for filtering terminal state ##
term_cond=np.zeros(state_dim)
term_cond[0]=1     # 1 if you want to fix x(t+T)[0]=x_ref[0], if not then 0
term_cond[1]=1     # 1 if you want to fix x(t+T)[1]=x_ref[1], if not then 0
term_cond[2]=0     # 1 if you want to fix x(t+T)[2]=x_ref[2], if not then 0
term_cond[3]=0     # 1 if you want to fix x(t+T)[3]=x_ref[3], if not then 0




#######################
## Two-link arm      ##
#######################
## system parameters ##
#######################
m1=0.25#[kg]
l1=0.5#[m]
I1=0.0125#[kgm^2]
m2=0.25#[kg]
l2=0.5#[m]
I2=0.0125#[kgm^2]

######################
##  state equation ###
######################
def plant(t, x, u):

    dxdt = np.zeros(x.shape[0])
    Mat=np.zeros([2,2])
    Mat[0,0]=(m1/4+m2)*l1*l1+I1
    Mat[0,1]=m2*l1*l2/2*np.cos(x[0]-x[1])
    Mat[1,0]=Mat[0,1]
    Mat[1,1]= m2/4*l2*l2+I2

    C=np.zeros(2)
    C[0]=-m2*l1*l2/2*x[3]**2*np.sin(x[0]-x[1])+u[0]-u[1]
    C[1]= m2*l1*l2/2*x[2]**2*np.sin(x[0]-x[1])+u[1]
    
    tmp=np.linalg.solve(Mat,C)

    dxdt[0]=x[2]
    dxdt[1]=x[3]
    dxdt[2]=tmp[0]
    dxdt[3]=tmp[1]

    return dxdt
    




##################################
##            u^(2)=0          ###
##################################
def dUdt_2nd_order(U):
    dUdt=np.zeros(U.shape[0])
    dUdt[0]=U[2]
    dUdt[1]=U[3]
    dUdt[2]=0
    dUdt[3]=0
    return dUdt

##################################
##            u^(3)=0          ###
##################################
def dUdt_3rd_order(U):
    dUdt=np.zeros(U.shape[0])
    dUdt[0]=U[2]
    dUdt[1]=U[3]
    dUdt[2]=U[4]
    dUdt[3]=U[5]
    dUdt[4]=0
    dUdt[5]=0
    return dUdt

##################################
##            u^(K)=0          ###
##################################
class dUdt:
    def __init__(self, diff_odr, input_dim):
        self.diff_odr=diff_odr
        self.input_dim=input_dim

    def Kth_order(self, U):
        dUdt=np.zeros(U.shape[0])
        diff_order=self.diff_odr
        input_dim=self.input_dim
        for i in range(diff_order-1):
            for j in range(input_dim):
                dUdt[input_dim*i+j]=U[input_dim+input_dim*i+j]
        for i in range(input_dim):
            dUdt[input_dim*diff_order-1-i]=0

        return dUdt



















































#######################
## Ctrler definition ##
#######################
UFunc=dUdt(diff_order, input_dim)
#CGinv=C_Ginv(plant, dUdt_2nd_order, input_dim)
#CGinv=C_Ginv(plant, dUdt_3rd_order, input_dim)
Ctrler=C_Ginv(plant, UFunc.Kth_order, input_dim, term_cond)



#################
##  state :  x ##
#################
x=np.zeros([max_iter+1,state_dim])
x[0,:]=x_init



##############
## input: u ##
##############
u=np.zeros([max_iter+1,input_dim])


##################################################################
## input state:  U:=[u, u', u", u^(3),...,u^(diff_order-1)]^T   ##
##################################################################
U_init =np.zeros(diff_order*input_dim)







#############
## time: t ##
#############
t=np.zeros(max_iter+1)
t[0]=t0


#################################
## list for graph of calc_time ##
#################################
t_list=[]
calc_time_list=[]


###################################
## variable for timing calc_time ##
###################################
t_start=None
t_end=None







############################
############################
############################
### Start               ####
### MPC simulation      ####
############################
############################
############################
############################

###################
### start loop ####
#####################################
### 0th loop of MPC computation  ####
#####################################
t_start = time.time()
u[0] = Ctrler.u_init(x[0], x_ref, t[0], T, U_init, N, tolerance=tol, max_iter=max_iterations, k=k)
t_end = time.time()
calc_time_list.append(t_end-t_start)
t_list.append(t[0])



## displaying some results ##
print('t:{:.3g}'.format(t[0]),'[s] | u[',0,'] =',u[0])
print('   F(t,x,U):evaluation_count =',Ctrler.F.eval_count,'times')
print('   calc time ={:.4g}'.format(t_end-t_start),'[s]')
print('   N =',N,', Horizon=',T,'[s]')
F=Ctrler.F(t[0],x[0],Ctrler.U)
print('   |F(t,x,U)|=',np.linalg.norm(F))





#####################################
### time evolution of real plant ####
#####################################
x[1] = x[0] + plant(t[0],x[0],u[0]) * dt
t[1] = t[0] + dt
    

## printing how close we are to the target state ##
print('|x[0]-x_ref| =',np.linalg.norm(Ctrler.F.TermFilter@(x[1]-x_ref)))
print()



## resetting count of evaluation of F(t,x,U) ##
Ctrler.F.eval_count = 0

############################
### loops 1 ~ max_iter  ####
############################
u_discrete = u[0]
t_prev = t[0]
for i in range(1,max_iter):
    if SamplingT - delta < t[i]-t_prev and\
       t[i]-t_prev < SamplingT + delta:
        ############################
        ### MPC computation     ####
        ############################
        t_start = time.time()
        u_discrete = Ctrler.u(x[i],x_ref,t[i],T,Ctrler.U,N,dt,zeta)
        t_end = time.time()
        calc_time_list.append(t_end-t_start)
        t_list.append(t[i])


        ## displaying some results ##
        print('t:{:.5g}'.format(t[i]),'[s] | u[',i,'] =',u_discrete)
        print('   F(t,x,U):evaluation_count =',Ctrler.F.eval_count,'times')
        print('   calc time ={:.4g}'.format(t_end-t_start),'[s]')
        print('   N =',N,', Horizon=',T,'[s]')
        F=Ctrler.F(t[i],x[i],Ctrler.U)
        print('   |F(t,x,U)|=',np.linalg.norm(F))
        print('   |x[',i,']-x_ref|=',np.linalg.norm(x[i]-x_ref))
        print()

        t_prev = t[i]




    #####################################
    ### time evolution of real plant ####
    #####################################
    u[i] = u_discrete
    x[i+1]=x[i]+plant(t[i],x[i], u[i])*dt
    t[i+1]=t[i]+dt
    



    ## resetting count of evaluation of F(t,x,U) ##
    Ctrler.F.eval_count=0


## displaying calculation time results ##
calc_time_list=np.array(calc_time_list)
max_index=np.argmax(calc_time_list)
min_index=np.argmin(calc_time_list)
avg_calc_time=np.mean(calc_time_list[1:])

print('longest   calc time(@i=',max_index,end='')
print('|t={:.3g}'.format(t[max_index]),end='')
print(')={:.4g}'.format(calc_time_list[max_index]),'[sec]')

print('shortest  calc time(@i=',min_index,end='')
print('|t={:.3g}'.format(t[min_index]),end='')
print(')={:.4g}'.format(calc_time_list[min_index]),'[sec]')

print('Average calculation time:',avg_calc_time,'[sec]')
print('Horizon T=',T,', Sampling period =',SamplingT)
print('N=',N,', diff_order=',diff_order,', input_dim=',input_dim)












































fig = plt.figure()

plt.plot(t_list,calc_time_list)
plt.axhline(y=SamplingT, xmin=0.0, xmax=Tf, linestyle='dotted')
plt.xlabel('time[s]', fontsize=14)
plt.ylabel('Computation time[s]', fontsize=14)

plt.grid()
#fig.savefig(file_name+'CalcTime.png', pad_inches=0.0)
plt.show()













fig = plt.figure()

plt.plot(t, x[:,0], label='theta1', marker='',linestyle='solid')
plt.plot(t, x[:,1], label='theta2', marker='',linestyle='dashed')
plt.axhline(y=x_ref[0], xmin=0.0, xmax=Tf, linestyle='dotted')
plt.axhline(y=x_ref[1], xmin=0.0, xmax=Tf, linestyle='dotted')



plt.ylabel('[rad]', fontsize=14)
plt.xlabel('time[s]', fontsize=14)

plt.grid()
plt.legend()
#fig.savefig(file_name+'Theta.png', pad_inches=0.0)
plt.show()











fig = plt.figure()

plt.plot(t, u[:,0], label='u1')
plt.plot(t, u[:,1], label='u2',linestyle='dashed')
plt.xlabel('time[s]', fontsize=14)
plt.ylabel('[Nm]', fontsize=14)

plt.grid()
plt.legend()
#fig.savefig(file_name+'Inputs.png', pad_inches=0.0)
plt.show()


