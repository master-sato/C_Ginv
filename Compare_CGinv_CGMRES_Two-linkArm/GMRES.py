'''
Generalized Minimum RESidual (GMRES) methods

Made in Jan. 2022

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



class GMRES:
    def __init__(self):
        self.iter=0
        self.FuncEvalCnt=0

    def GeneratePlaneRotation(self,dx, dy, cs, sn):
        if dy == 0.0:
            cs = 1.0
            sn = 0.0
        elif abs(dy) > abs(dx):
            temp = dx / dy
            sn = 1.0 / np.sqrt( 1.0 + temp*temp )
            cs = temp * sn
        else:
            temp = dy / dx
            cs = 1.0 / np.sqrt( 1.0 + temp*temp )
            sn = temp * cs

        return cs, sn



    def ApplyPlaneRotation(self,dx,dy,cs,sn):
        temp  =  cs * dx + sn * dy
        d_y = -sn * dx + cs * dy
        d_x = temp
        return d_x, d_y



    def Update(self,x,k,h, s,v):
        y=s
        for i in reversed(range(0,k+1)):
            y[i] =y[i]/ h[i,i]
            for j in reversed(range(0,i)):
                y[j] =y[j] -h[j,i] * y[i]

        for j in range(0,k+1):
            x =x+ v[j] * y[j]

        return x





    def GMRES(self,A, x, b, max_iter, tol):
        resid=0
        i, j = 1,1
        m=A.shape[0];

        s=np.zeros([m+1,1])
        cs=np.zeros([m+1,1])
        sn=np.zeros([m+1,1])

        w=None
  
        normb = np.linalg.norm(b)

        r = b - A @ x
        beta = np.linalg.norm(r)
  
        if normb == 0.0:
            normb = 1
  
        resid = beta / normb
        if resid <= tol:
            tol = resid
            max_iter = 0
            self.iter=max_iter
            return 0

        H=np.zeros([m+1,m+1])
        v=np.zeros([m+1,A.shape[0]])


        while j <= max_iter:
            v[0] = r * (1.0 / beta)
            s[0] = beta;
    
            for i in range(0,m):
                w = A @ v[i];
                for k in range(0,i+1):
                    H[k, i] = np.dot(w,v[k])
                    w =w-  v[k]*H[k,i]
                H[i+1, i] = np.linalg.norm(w)
                v[i+1] = w * (1.0 / H[i+1,i])



                for k in range(0,i):
                    H[k,i],H[k+1,i]=self.ApplyPlaneRotation(H[k,i],H[k+1,i],cs[k],sn[k])


                cs[i],sn[i]=self.GeneratePlaneRotation(H[i,i],H[i+1,i],cs[i],sn[i])
                H[i,i], H[i+1,i]=self.ApplyPlaneRotation(H[i,i],H[i+1,i],cs[i],sn[i])
                s[i], s[i+1]=self.ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i])





                resid = abs(s[i+1])
                if (resid / normb) < tol:
                    x=self.Update(x, i, H, s, v)
                    tol = resid
                    max_iter = j
                    self.iter=max_iter
                    return x

                if i<m or j <= max_iter:
                    j=j+1
                else:
                    break

            x=self.Update(x, m - 1, H, s, v)
            r = (b - A @ x)
            beta = np.linalg.norm(r)
            resid = beta / normb
            if resid < tol:
                tol = resid
                max_iter = j
                self.iter=max_iter
                return 0
  
        tol = resid
        return 0
























    def FDGMRES(self, Func, xnow,vnow,h, b, max_iter, tol, Fnow=None):
        resid=0
        i, j = 1,1
        if isinstance(Fnow, np.ndarray):
            pass
        elif Fnow==None:
            Fnow=Func(xnow)
            self.FuncEvalCnt=self.FuncEvalCnt+1
        else:
            print('Error inside FDGMRES(): Fnow must be None or a numpy.ndarray.')
            exit(1)



        self.FuncEvalCnt=self.FuncEvalCnt+1
    
        m=Fnow.shape[0]+1;

        s=np.zeros([m+1,1])
        cs=np.zeros([m+1,1])
        sn=np.zeros([m+1,1])

        w=None
  
        normb = np.linalg.norm(b)

        r = b - (Func(xnow+vnow*h)-Fnow)/h
        self.FuncEvalCnt=self.FuncEvalCnt+1

        beta = np.linalg.norm(r)
  
        if normb == 0.0:
            normb = 1
  
        resid = beta / normb
        if resid <= tol:
            tol = resid
            max_iter = 0
            self.iter=max_iter
            return 0

        H=np.zeros([m+1,m+1])
        v=np.zeros([m+1,Fnow.shape[0]])


        while j <= max_iter:
            v[0] = r * (1.0 / beta)    #// ??? r / beta
            s[0] = beta
    
            for i in range(0,m):
                w = (Func(xnow+v[i]*h)-Fnow)/h;
                self.FuncEvalCnt=self.FuncEvalCnt+1
                for k in range(0,i+1):
                    H[k, i] = np.dot(w,v[k])
                    w =w-  v[k]*H[k,i]
                H[i+1, i] = np.linalg.norm(w)
                v[i+1] = w * (1.0 / H[i+1,i])



                for k in range(0,i):
                    H[k,i],H[k+1,i]=self.ApplyPlaneRotation(H[k,i],H[k+1,i],cs[k],sn[k])


                cs[i],sn[i]=self.GeneratePlaneRotation(H[i,i],H[i+1,i],cs[i],sn[i])
                H[i,i], H[i+1,i]=self.ApplyPlaneRotation(H[i,i],H[i+1,i],cs[i],sn[i])
                s[i], s[i+1]=self.ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i])





                resid = abs(s[i+1])
                if (resid / normb) < tol:
                    vnow=self.Update(vnow, i, H, s, v)
                    tol = resid
                    max_iter = j
                    self.iter=max_iter
                    return vnow

                if i<m or j <= max_iter:
                    j=j+1
                else:
                    break

            vnow=self.Update(vnow, m - 1, H, s, v)
            r = b-(Func(xnow+vnow*h)-Fnow)/h
            self.FuncEvalCnt=self.FuncEvalCnt+1
            beta = np.linalg.norm(r)
            resid = beta / normb
            if resid < tol:
                tol = resid
                max_iter = j
                self.iter=max_iter
                return x
  
        tol = resid
        return 0































    def JFNewtonGMRES(self,function, x,tolerance,h,max_iter=10, k=1.0):
        x_now=x.copy()
        col=len(x_now)

        i=0

        for i in range(max_iter):
            Fnow = function(x_now)

            if np.linalg.norm(Fnow)<=tolerance:
                print('HIT! inside Jacobian-freeNewtonGMRES i=',i,'norm(F)=',np.linalg.norm(Fnow))
                break
            else:
                print('     inside Jacobian-freeNewtonGMRES i=',i,'norm(F)=',np.linalg.norm(Fnow))



            v_now=np.zeros(x_now.shape[0])
  
            b=Fnow
            delta=self.FDGMRES(function, x_now,v_now,h,b,max_iter,tolerance,Fnow=Fnow)

            x_now = x_now - k*delta

        if i==max_iter-1:
            print('NO HIT inside Jacobian-freeNewtonGMRES max_iter-1=',max_iter-1)
            print('       norm(F)=',np.linalg.norm(Fnow))

        return x_now


