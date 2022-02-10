'''
Continuation/Generalized-inverse (G/Ginv) method

Made in Jan. 2022 ver.0.1


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





def Gauss_Newton(function, x, h, tolerance, max_iter=10, k=1.0):
    xnow=x
    col=len(xnow)

    i=0

    for i in range(max_iter):
        F = function(xnow)

        if np.linalg.norm(F)<=tolerance:
            print('HIT! inside Gauss_Newton i=',i,'norm(F)=',np.linalg.norm(F))
            break
        else:
            print('     inside Gauss_Newton i=',i,'norm(F)=',np.linalg.norm(F))

        J=Jacobian(function,xnow,h,F)


        delta=None
        if J.shape[0]==J.shape[1]:
            if np.linalg.det(J)>1e-2:
                delta=np.linalg.solve(J,F)
            else:
                delta = np.linalg.pinv(J).dot(F)
        else:
            delta = np.linalg.pinv(J).dot(F)
        xnow = xnow - k*delta


    if i==max_iter:
        print('NO HIT inside Gauss_Newton max_iter-1=',max_iter-1)
        print('       norm(F)=',np.linalg.norm(F))

    return xnow























class TPBVP:
    def __init__(self, xFunc,uFunc, u_dim, term_cond=0):
        self.dxdt=xFunc     # dx/dt=f(t,x,u) (State equation)
        self.dudt=uFunc     # u^(k)=0  (Input dynamics)
        self.u_dim=u_dim    # Input dimension
        self.N=None         # Integration steps (number of grids)
        self.T=None         # Prediction horizon
        self.t=None         # Current time
        self.dt=None        # time step for MPC computation
        self.x=None         # Current state
        self.x_ref=None     # Target state

        self.eval_count=0   # Counter for number of times self.__call__() is called


        self.TermFilter=None
        if isinstance(term_cond,np.ndarray):
            self.TermFilter=self.FilterMatrix(term_cond)
        else:
            self.TermFilter=term_cond


    def set_params(self, x,x_ref, t, T, N):
        self.N=N
        self.T=T
        self.t=t
        self.dt=self.T/self.N
        self.x=x
        self.x_ref=x_ref

    def make_u_from_U(self,U):
        ans=np.zeros(self.u_dim)
        for i in range(self.u_dim):
            ans[i]=U[i]
        return ans


    def FilterMatrix(self,cond_vec):
        count=0
        for i in range(cond_vec.shape[0]):
            if cond_vec[i]!=0:
                count+=1
        ans=np.zeros([count,cond_vec.shape[0]])

        count=0
        for i in range(ans.shape[0]):
            for j in range(count,ans.shape[1]):
                if cond_vec[j]!=0:
                    ans[i,j]=cond_vec[j]
                    count+=1
                    break

        return ans


    def __call__(self, t, xnow, Unow):
        self.eval_count+=1
    
        for i in range(self.N):
            unow=self.make_u_from_U(Unow)
            xnext=xnow+self.dxdt(t,xnow,unow)*self.dt
            Unext=Unow+self.dudt(Unow)*self.dt
                
            t=t+self.dt
            xnow=xnext
            Unow=Unext

        dx=xnow-self.x_ref
        ans=dx
        if isinstance(self.TermFilter,np.ndarray):
            ans=self.TermFilter@dx

        return ans


    def func_of_U(self, U):
        return self.__call__(self.t, self.x, U)
        














class C_Ginv:
    def __init__(self, xFunc, uFunc, u_dim, term_cond=0):
        self.F=TPBVP(xFunc, uFunc, u_dim, term_cond)
        self.U=None


    def u_init(self, xnow, x_ref, t, T, U_guess, N, tolerance=1e-5, max_iter=50, k=1.0):
        self.F.set_params( xnow, x_ref, t, T, N)
        
        U_init=Gauss_Newton(self.F.func_of_U, U_guess, self.F.dt, tolerance, max_iter, k)

        self.U=U_init

        u_init=self.F.make_u_from_U(self.U)

        return u_init


    def u(self, xnow, x_ref, t, T, Unow, N,dt,zeta):
        self.F.set_params( xnow, x_ref, t, T, N)
        h=self.F.dt
        Fnow=self.F(t,xnow,Unow)
        unow=self.F.make_u_from_U(Unow)


        dxdt=self.F.dxdt(t, xnow, unow)
        F_xnow_dxnow__F_t = (self.F(t+h, xnow+dxdt*h, Unow)-Fnow)/h
        b = -F_xnow_dxnow__F_t-zeta*Fnow


        J=Jacobian(self.F.func_of_U, Unow, h,Fnow)
        dUdt=np.linalg.pinv(J)@(b)


        U_next=Unow+dUdt*dt

        self.U=U_next

        u_next=self.F.make_u_from_U(self.U)
        


        return u_next


