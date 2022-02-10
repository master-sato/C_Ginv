'''
Example of Continuation/GMRES (G/GMRES) method
(Two-link arm)

Made in Feb. 2022 ver. 0.1

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
from CGMRES import C_GMRES
import time









###########################
## simulation parameters ##
###########################
state_dim=4    # state dimension
input_dim=2    # input dimension

t0=0.0         # initial time [s]
T=0.2         # Horizon [s]
N=2           # Integration steps within the MPC computation

dt=0.025         # Sampling time [s]
Tf=5           # Simulation time [s]
iter=int((Tf-t0)/dt)   # iteration of simulation (for loop iteration)
zeta=1/dt      # parameter for C/GMRES


## parameters for GMRES and Newton type methods ##
tol = 1e-5           # terminates iteration when norm(Func) < tol
max_iter_Newton = 10  # maximum iteration of Gauss-Newton method
max_iter_FDGMRES =3  # maximum iteration of Gauss-Newton method
k = 1                # damping coefficient inside Gauss-Newton method


#################################
## file_name for saving graphs ##
#################################
file_name='CGMRES_DoublePen_T'+str(T)+'N'+str(N)+'dt'+str(dt)






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


############################################################
## weight parameters of nonlinear optimal control problem ##
## J= (x(t+T)-x_ref)^T*S*(x(t+T)-x_ref)/2                 ##
##   +Integral[x^T*Q*x/2+u^T*R*u/2]                       ##
Q=np.eye(state_dim, state_dim)
R=np.eye(input_dim, input_dim)
S=np.eye(state_dim, state_dim)

Q[0,0]=40
Q[1,1]=20
Q[2,2]=0.01
Q[3,3]=0.01

R[0,0]=1
R[1,1]=1


S[0,0]=4
S[1,1]=2
S[2,2]=0.001
S[3,3]=0.001







#######################
## Double Pendulum   ##
#######################
## system parameters ##
#######################
m1=0.25#[kg]
l1=0.5#[m]
I1=0.0125#[kgm^2]
m2=0.25#[kg]
l2=0.5#[m]
I2=0.0125#[kgm^2]



################################
##  state eq of Two-link arm ###
################################
def plant(t, x, u):

    dxdt = np.zeros(x.shape[0])
    Mat=np.zeros([2,2])
    Mat[0,0]=(m1/4+m2)*l1*l1+I1
    Mat[0,1]=m2*l1*l2/2*np.cos(x[0]-x[1])
    Mat[1,0]=Mat[0,1]
    Mat[1,1]= m2/4*l2*l2+I2

    C=np.zeros(2)
    C[0]=-m2*l1*l2/2*x[3]*x[3]*np.sin(x[0]-x[1])+u[0]-u[1]
    C[1]= m2*l1*l2/2*x[2]*x[2]*np.sin(x[0]-x[1])+u[1]
    
    tmp=np.linalg.solve(Mat,C)

    dxdt[0]=x[2]
    dxdt[1]=x[3]
    dxdt[2]=tmp[0]
    dxdt[3]=tmp[1]

    return dxdt
    



    































###########################
## Controller definition ##
###########################
Ctrler=C_GMRES(plant,state_dim, input_dim,Q,R,S)



#################
##  state :  x ##
#################
x=np.zeros([iter+1,state_dim])
x[0,0]=x_init[0]
x[0,1]=x_init[1]

##############
## input: u ##
##############
u=np.zeros([iter,input_dim])


##################################################################
## input state:  U:=[u[0], u[1],...,u[N-1]]^T   ##
##################################################################
U_init =np.zeros(N*input_dim)





#############
## time: t ##
#############
t=np.zeros(iter+1)
t[0]=t0



#################################
## list for graph of calc_time ##
#################################
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
u[0] = Ctrler.u_init(x[0], x_ref, t[0], T, U_init, N, tolerance=tol, max_iter=max_iter_Newton, k=k)
t_end = time.time()
calc_time_list.append(t_end-t_start)



## displaying some results ##
print('t:{:.4g}'.format(t[0]),'[s] | u[',0,'] =',u[0])
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
print('   |x[0]-x_ref| =',np.linalg.norm(x[1]-x_ref))
print()



## resetting count of evaluation of F(t,x,U) ##
Ctrler.F.eval_count = 0


#exit()

############################
### loops 1 ~ max_iter  ####
############################
for i in range(1,iter):
#for i in range(1,2):
    ############################
    ### MPC computation     ####
    ############################
    t_start = time.time()
    u[i] = Ctrler.u(x[i],x_ref,t[i],T,Ctrler.U,N, dt,zeta,tol, max_iter_FDGMRES)
    t_end = time.time()
    calc_time_list.append(t_end-t_start)

    ## displaying some results ##
    print('t:{:.4g}'.format(t[i]),'[s] | u[',i,'] =',u[i])
    print('   F(t,x,U):evaluation_count =',Ctrler.F.eval_count,'times')
    print('   calc time ={:.4g}'.format(t_end-t_start),'[s]')
    print('   N =',N,', Horizon=',T,'[s]')
    F=Ctrler.F(t[i],x[i],Ctrler.U)
    print('   |F(t,x,U)|=',np.linalg.norm(F))
    print('   |x[',i,']-x_ref|=',np.linalg.norm(x[i]-x_ref))
    print()




    #####################################
    ### time evolution of real plant ####
    #####################################
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
print('Horizon T=',T,', Sampling Time dt=',dt)
print('N=',N,', input_dim=',input_dim)
print('Q=')
print(Q)
print('R=')
print(R)
print('S=')
print(S)





















fig = plt.figure()

plt.plot(t[:-1],calc_time_list)
plt.axhline(y=dt, xmin=0.0, xmax=Tf, linestyle='dotted')
plt.xlabel('time[s]')
plt.ylabel('Computation time[s]')

plt.grid()
plt.legend()
#plt.axes().set_aspect('equal')
#fig.savefig(file_name+'CalcTime.png', pad_inches=0.0)
plt.show()






fig = plt.figure()

plt.plot(t, x[:,0], label='theta1', marker='',linestyle='solid')
plt.plot(t, x[:,1], label='theta2', marker='',linestyle='dashed')
plt.axhline(y=x_ref[0], xmin=0.0, xmax=Tf, linestyle='dotted')
plt.axhline(y=x_ref[1], xmin=0.0, xmax=Tf, linestyle='dotted')



plt.ylabel('[rad]')
plt.xlabel('time[s]')

plt.grid()
plt.legend()
#plt.axes().set_aspect('equal')
#fig.savefig(file_name+'Theta.png', pad_inches=0.0)
plt.show()













fig = plt.figure()

plt.plot(t, x[:,2], label='theta1_dot', marker='')
plt.plot(t, x[:,3], label='theta2_dot', marker='',linestyle='dashed')

plt.xlabel('time[s]')
plt.ylabel('[rad/s]')

plt.grid()
plt.legend()
#plt.axes().set_aspect('equal')
#fig.savefig(file_name+'Theta_dot.png', pad_inches=0.0)
plt.show()









fig = plt.figure()

#plt.plot(times, u_list[:,0], label='u1')
#plt.plot(times, u_list[:,1], label='u2',linestyle='dashed')
plt.plot(t[:-1], u[:,0], label='u1')
plt.plot(t[:-1], u[:,1], label='u2',linestyle='dashed')
plt.xlabel('time[s]')
plt.ylabel('[Nm]')

plt.grid()
plt.legend()
#plt.axes().set_aspect('equal')
plt.show()
#fig.savefig(file_name+'Inputs.png', pad_inches=0.0)








