'''
Example of Continuation/Generalized-inverse (G/Ginv) method

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
import time
from CGinv import C_Ginv








###########################
## simulation parameters ##
###########################
state_dim=3    # state dimension
input_dim=2    # input dimension
diff_order=2   # k of u^(k)=0

t0=0.0         # initial time [s]
T=0.15         # Horizon [s]
N=3            # Integration steps within the MPC computation
dt_MPC=T/3     # time step in the MPC computation

dt=0.005       # Sampling time [s]
Tf=30          # Simulation time [s]
max_iter=int((Tf-t0)/dt)+1   # iteration of simulation (for loop iteration)
zeta=1/dt_MPC  # parameter for C/Ginv
#zeta=20       # parameter for C/Ginv

## parameters for Gauss-Newton methods
tol = 1e-5           # terminates iteration when norm(Func) < tol
max_iterations = 15  # maximum iteration of Gauss-Newton method
k = 1                # damping coefficient inside Gauss-Newton method


## file_name for saving graphs ##
file_name='CGinv_TrajTracking_Two-wheeler_T'+str(T)+'N'+str(N)+'dt'+str(dt)+'DiffOrd'+str(diff_order)


## parameters of reference trajectory  ##
a=-1.0
A=15           # amplitude of cos curve
omega1=0.8
Radius=10      # radius of arc
omega2=0.5


####################
##  Initial state ##
####################
x_init=np.zeros(state_dim)
x_init[0]=-1          # Initial position x_init=[x,y,theta]^T=
x_init[1]=0           #             [-1[m], 0[m], 0[rad]]^T
x_init[2]=np.pi/180*(-0)



###########################
##  reference trajectory ##
###########################
x_ref=np.zeros([max_iter+1,state_dim])


boot_up=0
for i in range(1,x_ref.shape[0]-1):
    tau=i/max_iter*Tf

    eps=1e-5

    dxdt=np.zeros(state_dim)

    if tau<Tf/3:
        ## y=-a*tau ##
        dxdt[0]=1
        dxdt[1]=a

        x_ref[i+1]=x_ref[i]+dxdt*dt

    elif Tf/3<=tau and tau<2*Tf/3:
        ## y=A*sin(omega1*(tau-Tf/3)) ##
        dxdt[0]=1
        dxdt[1]=A*omega1*np.sin(omega1*(tau-Tf/3))
        x_ref[i+1]=x_ref[i]+dxdt*dt


    elif 2*Tf/3<=tau:
        ## circle ##
        dxdt[0]= Radius*omega2*np.sin(omega2*(tau-2*Tf/3))
        dxdt[1]=-Radius*omega2*np.cos(omega2*(tau-2*Tf/3))

        x_ref[i+1]=x_ref[i]+dxdt*dt



    ## to obtain x[3]:=theta of x_ref ##
    xnow=x_ref[i,0]
    ynow=x_ref[i,1]
    xprev=x_ref[i-1,0]
    yprev=x_ref[i-1,1]
    if abs(xnow-xprev)<eps:
        x_ref[i,2]=np.arctan((ynow-yprev)/(eps))+boot_up
    else:
        x_ref[i,2]=np.arctan((ynow-yprev)/(xnow-xprev))+boot_up
    if np.pi/2 - 3e-3 < x_ref[i,2]:
        boot_up=np.pi

    pass









## vector for filtering terminal state ##
term_cond=np.zeros(state_dim)
term_cond[0]=1     # 1 if you want to fix x(t+T)[0]=x_ref[0], if not then 0
term_cond[1]=1     # 1 if you want to fix x(t+T)[1]=x_ref[1], if not then 0
term_cond[2]=0     # 1 if you want to fix x(t+T)[2]=x_ref[2], if not then 0

#######################
## Double Pendulum   ##
#######################
## system parameters ##
#######################
D=2.0           # Wheel base of two-wheel vehicle
########################################
##  state func of Two Wheeler (bike) ###
########################################
def plant(t, x, u):
    out = np.zeros_like(x)
    out[0]=u[0]*np.cos(x[2])
    out[1]=u[0]*np.sin(x[2])
    out[2]=u[0]/D*np.tan(u[1])

    return out





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
#CGinv=C_Ginv(plant, dUdt_2nd_order, input_dim, term_cond)
#CGinv=C_Ginv(plant, dUdt_3rd_order, input_dim, term_cond)
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
u[0] = Ctrler.u_init(x[0], x_ref[0], t[0], T, U_init, N, tolerance=tol, max_iter=max_iterations, k=k)
t_end = time.time()
calc_time_list.append(t_end-t_start)



## displaying some results ##
print('t:{:.3g}'.format(t[0]),'[s] | u[',0,'] =',u[0])
print('   F(t,x,U):evaluation_count =',Ctrler.F.eval_count,'times')
print('   calc time ={:.4g}'.format(t_end-t_start),'[s]')
print('   N =',N,', Horizon=',T,'[s]')
F = Ctrler.F(t[0],x[0],Ctrler.U)
print('   |F(t,x,U)|=',np.linalg.norm(F))





#####################################
### time evolution of real plant ####
#####################################
x[1] = x[0] + plant(t[0],x[0],u[0]) * dt
t[1] = t[0] + dt
    

## printing how close we are to the target state ##
print('|x[0]-x_ref| =',np.linalg.norm(Ctrler.F.TermFilter@(x[1]-x_ref[1])))
print()



## resetting count of evaluation of F(t,x,U) ##
Ctrler.F.eval_count = 0

############################
### loops 1 ~ max_iter  ####
############################
for i in range(1,max_iter):
    ############################
    ### MPC computation     ####
    ############################
    t_start = time.time()
    u[i] = Ctrler.u(x[i],x_ref[i],t[i],T,Ctrler.U,N,dt,zeta)
    t_end = time.time()
    calc_time_list.append(t_end-t_start)

    ## displaying some results ##
    print('t:{:.3g}'.format(t[i]),'[s] | u[',i,'] =',u[i])
    print('   F(t,x,U):evaluation_count =',Ctrler.F.eval_count,'times')
    print('   calc time ={:.4g}'.format(t_end-t_start),'[s]')
    print('   N =',N,', Horizon=',T,'[s]')
    F = Ctrler.F(t[i],x[i],Ctrler.U)
    print('   |F(t,x,U)|=',np.linalg.norm(F))
    print('   |x[',i,']-x_ref|=',np.linalg.norm(x[i]-x_ref[i]))
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
print('dt_MPC=',dt_MPC)
print('N=',N,', diff_order=',diff_order,', input_dim=',input_dim)
















































fig = plt.figure()

plt.plot(t[:-1],1000*calc_time_list)
plt.axhline(y=1000*dt, xmin=0.0, xmax=Tf, linestyle='dotted')
plt.xlabel('Time[s]')
plt.ylabel('Computation time[ms]')

plt.yticks([0.0,0.5,1,2,3,4,5,6])
plt.grid()
#fig.savefig(file_name+'CalcTime.png', pad_inches=0.0)
plt.show()






fig = plt.figure()

plt.plot(t, x[:,0], label='x', marker='',linestyle='solid')
plt.plot(t, x[:,1], label='y', marker='',linestyle='dashed')
plt.plot(t, x[:,2], label='theta', marker='',linestyle='dashdot')
plt.plot(t, x_ref[:,0], label='x_ref', marker='',linestyle='solid')
plt.plot(t[:-1], x_ref[:-1,1], label='y_ref', marker='',linestyle='dashed')
plt.plot(t[:-1], x_ref[:-1,2], label='theta_ref', marker='',linestyle='dotted')
#plt.axhline(y=x_ref[0], xmin=0.0, xmax=Tf, linestyle='dotted')
#plt.axhline(y=x_ref[1], xmin=0.0, xmax=Tf, linestyle='dotted')



plt.xlabel('Time[s]')

plt.grid()
plt.legend()
#fig.savefig(file_name+'_States.png', pad_inches=0.0)
plt.show()






















fig = plt.figure()

plt.plot(t[:-1], u[:-1,0], label='v [m/s]')
plt.plot(t[:-1], u[:-1,1], label='delta [rad]',linestyle='dashed')
plt.xlabel('Time[s]')

plt.grid()
plt.legend()
plt.show()
#fig.savefig(file_name+'_Inputs.png', pad_inches=0.0)











import matplotlib.patches as patch

fig = plt.figure()

plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.9)
#plt.rcParams['figure.subplot.left'] = 0.15

#plt.plot(x_list[:,0], x_list[:,1], label='car')
plt.plot(x_ref[:,0], x_ref[:,1], label='reference  trajectory',linestyle='dotted',color='blue')
plt.plot(x[:,0], x[:,1], label='two-wheel vehicle',color='red')
#plt.plot(times, u_list[:,0], label='u x 0.1')
plt.xlabel('x[m]',fontsize=11)
plt.ylabel('y[m]',fontsize=11)
#plt.legend(loc='upper right')

#plt.xlim([-5, 42.5])
#plt.ylim([-12.5, 22.5])

X=x[-1,0]
Y=x[-1,1]
theta=x[-1,2]
L=6
dX=L*np.cos(theta)
dY=L*np.sin(theta)
arrow= patch.Arrow(x=X, y=Y, dx=dX, dy=dY,width=3.0,color='black')
plt.axes().add_patch(arrow)


#plt.axes().set_aspect('equal')
plt.grid()
#plt.legend()
plt.legend(prop={'size':10}, loc='lower right')
plt.axes().set_aspect('equal')
plt.show()

#fig.savefig(file_name+'[x,y].png', pad_inches=0.0)



