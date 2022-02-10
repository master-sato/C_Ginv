'''
Continuation/Generalized Minumim RESidual (C/GMRES) method


This source code is implemented based on [1].
[1] Ohtsuka, T. ;
    A Continuation/GMRES method for fast computation of nonlinear receding horizon control;
    Automatica;
    Vol. 40, No. 4, pp. 563 - 574 (2004).




Made in Jan. 2022 ver. 0.1


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
from GMRES import GMRES



#'''
def Jacobian(Func,x,h, F_now=None):
    xnow=x
    if type(F_now)==np.ndarray:
        Fnow=F_now
    else:
        Fnow=Func(xnow)
    row = Fnow.shape[0]
    col = xnow.shape[0]
    J=np.zeros([row,col])
    for i in range(col):
        dx=np.zeros(col)
        dx[i]=h
        J[:,i]=Func(xnow+dx)- Fnow
    J=J/h
    
    return J































class BigF:
    def __init__(self, xFunc, state_dim, u_dim, Q,R,S):
        self.dxdt=xFunc                   # state equation
        self.u_dim=u_dim                  # input dimension
        self.state_dim=state_dim          # state dimension
        self.N=None                       # Integration steps (number of grids)
        self.T=None                       # Prediction horizon
        self.t=None                       # Current time
        self.dt=None                      # Time step for MPC computation
        self.x=None                       # Current state
        self.x_ref=None                   # Target state

        '''
        self.c_func=None
        self.dummy_dim=0
        '''

        ########################################
        ## J= x(t+T)^T*S*x(t+T)/2             ##
        ##   +Int[x^T*Q*x/2+u^T*R*u/2]dt      ##
        ########################################
        self.Q=Q
        self.R=R
        self.S=S

        self.dfdx=None                    # Analytical df/dx
        self.analytical_dfdx_flag=0       # 1 if we have analytical df/dx, 0 if not.
        self.dfdu=None                    # Analytical df/du
        self.analytical_dfdu_flag=0       # 1 if we have analytical df/du, 0 if not.

        self.eval_count=0                 # Counter for number of times self.__call__() is called
        

    def set_analytical_dfdx(self, dfdx):
        self.dfdx=dfdx
        self.analytical_dfdx_flag=1

    def set_analytical_dfdu(self, dfdu):
        self.dfdu=dfdu
        self.analytical_dfdu_flag=1



    def set_constraint(self,c_func, dummy_dim):
        self.c_func=c_func
        self.dummy_dim=dummy_dim

    def set_params(self, x,x_ref,U, t, T, N):
        self.N=N
        self.T=T
        self.t=t
        self.dt=self.T/self.N
        self.x_now=x
        self.x_ref=x_ref
        self.U_now=U




    def __call__(self, tnow, xnow, Unow):
        self.eval_count+=1
        
        x = np.zeros([self.N+1,self.state_dim])        # state
        p = np.zeros([self.N+1,self.state_dim])        # Co-state
        u = np.zeros([self.N, self.u_dim])             # input
        for i in range(self.N):
            for j in range(self.u_dim):
                u[i,j]=Unow[i*self.u_dim+j]

        x[0] = xnow
        t = tnow


        for i in range(self.N):
            dxdt=self.dxdt(t,x[i],u[i])
            x[i+1]=x[i]+dxdt*self.dt
            t=t+self.dt


        p[-1] = self.S.T@(x[-1]-self.x_ref)
        t=tnow+self.T
        dfdx = None
        for i in reversed(range(1,self.N+1)):
            if self.analytical_dfdx_flag==1:
                dfdx = self.dfdx(t,x[i],u[i-1])
            else:
                self.set_u_tmp(u[i-1])
                dfdx = Jacobian(self.xfunc_of_x,x[i],self.dt)

            dpdt=-(self.Q.T@x[i]+dfdx.T@p[i])
            p[i-1]=p[i]-dpdt*self.dt
            t=t-self.dt




        t=tnow
        dHdu=np.zeros(self.N*self.u_dim)
        dfdu=None
        for i in range(0,self.N):
            if self.analytical_dfdu_flag==1:
                dfdu = self.dfdu(t,x[i],u[i])
            else:
                self.set_x_tmp(x[i])
                dfdu = Jacobian(self.xfunc_of_u,u[i],self.dt)

            tmp= self.R.T@u[i]+dfdu.T@p[i]

            for j in range(self.u_dim):
                dHdu[i*self.u_dim+j]=tmp[j]
            t=t+self.dt

        return dHdu











    def func_of_U(self, U):
        return self.__call__(self.t, self.x_now, U)
        


    def set_x_tmp(self,x):
        self.x_tmp=x

    def set_u_tmp(self,u):
        self.u_tmp=u

    def xfunc_of_x(self, x):
        return self.dxdt(self.t,x, self.u_tmp )

    def xfunc_of_u(self, u):
        return self.dxdt(self.t,self.x_tmp, u )










class C_GMRES:
    def __init__(self, xFunc, state_dim, u_dim, Q,R,S):
        self.F=BigF(xFunc, state_dim, u_dim,Q,R,S)
        self.U=None


    def make_u_from_U(self,U):
        ans=np.zeros(self.F.u_dim)
        for i in range(self.F.u_dim):
            ans[i]=U[i]
        return ans

    def set_analytical_dfdx(self, dfdx):
        self.F.set_analytical_dfdx(dfdx)

    def set_analytical_dfdu(self, dfdu):
        self.F.set_analytical_dfdu(dfdu)


    def u_init(self, xnow, x_ref, t, T, U_guess, N, tolerance=1e-5, max_iter=50, k=1.0):
    

        self.F.set_params( xnow, x_ref, U_guess, t, T, N)
        

        Solver=GMRES()
        U_init=Solver.JFNewtonGMRES(self.F.func_of_U, U_guess,tolerance,self.F.dt,max_iter, k)

        self.U=U_init

        u_init=self.make_u_from_U(self.U)
        return u_init



    def u(self, xnow, x_ref, t, T, Unow, N, dt,zeta, tolerance=1e-5, max_iter=50):
        U_now=Unow
        self.F.set_params( xnow, x_ref,Unow, t, T, N)
        h=self.F.dt
        Fnow=self.F(t,xnow,Unow)
        unow=self.make_u_from_U(Unow)

        dxdt=self.F.dxdt(t, xnow, unow)
        F_xnow_dxnow__F_t=(self.F(t+h, xnow+dxdt*h, U_now)-Fnow)/h

        b = -F_xnow_dxnow__F_t-zeta*Fnow

        
        Solver=GMRES()
        dUdt=np.zeros(Unow.shape[0])
        dUdt=Solver.FDGMRES(self.F.func_of_U, Unow,dUdt,h, b, max_iter, tolerance, Fnow=Fnow)

        U_next=Unow+dUdt*dt

        self.U=U_next

        u_next=self.make_u_from_U(self.U)
        return u_next


