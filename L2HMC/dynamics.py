import torch
import torch.nn as nn
import numpy as np
import pdb

def safe_exp(x, name=None):
    return torch.exp(x)
    

torchType = torch.float32
numpyType = np.float32
# x_dim = 32
# n_samples = 1
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# M = torch.eye(x_dim) + 0j # (Nt*Nx) x (Nt*Nx) = [x_dim x x_dim] identity matrix
# M_all= [M for i in range(n_samples)]  #[n_samples,[x_dim,x_dim]]  Fermion matrix


class Dynamics(nn.Module):
    """
    The main class that describes modidied HMC dynamics
    """
    def __init__(self,
                 x_dim,
                 deltaU,
                 expk,
                 Nt,
                 T=25,
                 eps=0.1,
                 n_samples = 1,
                 hmc=False,
                 net_factory=None,
                 eps_trainable=True,
                 use_temperature=False,
                 ):
        super(Dynamics, self).__init__()
       
        self.x_dim = x_dim                               # dimensionality of input x
        self.Nt = Nt                                     #time steps
        self.deltaU = deltaU                             #reduced U
        self.expk = expk                                 #exponential Hopping matrix
        self.n_samples = n_samples
        self.use_temperature = use_temperature           # whether we use temperature or not
        self.temperature = torch.tensor(5.0, dtype=torchType)  # temperature value
        self.M = torch.eye(x_dim) + 0j # (Nt*Nx) x (Nt*Nx) = [x_dim x x_dim] identity matrix
        
        if not hmc:
            self.alpha = nn.Parameter(torch.log(torch.tensor(eps,dtype=torchType)),requires_grad=eps_trainable)
        else:
            self.alpha = torch.log(torch.tensor(eps, dtype=torchType))  #eps in leapfrog    

        
        self.hmc = hmc  # whether we use hmc or not

        # if HMC we just return all zeros
        if hmc:
            z = lambda x, *args, **kwargs: tf.zeros_like(x)
            self.XNet = lambda inp: [torch.zeros_like(inp[0]) for t in range(3)]
            self.VNet = lambda inp: [torch.zeros_like(inp[0]) for t in range(3)]
        else:
            self.XNet = net_factory(x_dim, scope='XNet', factor=2.0)  # net_factory is just a NN
            self.VNet = net_factory(x_dim, scope='VNet', factor=1.0)
            
        self.T = T
        self._init_mask()
        self.T = torch.tensor(T, dtype=torchType)  # number of iteration(Nmd) for forward/backward procedure during training
        
        if self.use_temperature:
            self.Temp = torch.tensor(self.temperature, dtype=torchType)
        else:
            self.Temp = torch.tensor(1.0, dtype=torchType)

    def _init_mask(self):  # just set a half of components to zero, and a half to one
        
        mask_per_step = []
        
        for t in range(self.T):
            ind = np.random.permutation(torch.arange(self.x_dim))[:int(self.x_dim / 2)]
            m = np.zeros((self.x_dim,))
            m[ind] = 1
            mask_per_step.append(m)
        self.mask = torch.tensor(np.stack(mask_per_step), dtype=torchType)

    def _get_mask(self, step):
        m = self.mask[step.type(torch.int32), ...]  # , torch.tensor(step, dtype=torchType))
        return m, 1. - m

    def _format_time(self, t, tile=1):
        trig_t = torch.cat([
            torch.cos(2 * np.pi * t / self.T)[..., None],
            torch.sin(2 * np.pi * t / self.T)[..., None],
        ])
        out = trig_t.repeat(tile, 1)
        assert out.shape == (tile, 2), 'in _format_time'
        return out                                             # outputs tensor of size tile x 2

    
    
    def Mphi(self,x):
        """
        Function that updates the Fermion matrix

        INPUT:
                x - phi field (array)
                nt - number of time steps

        OUTPUT:
                M - fermion matrix (update M) [Nt*Nx ,Nt*Nx]
                M_all - array of fermion matrix [num_samples X Nt*Nx X Nt*Nx]

                This uses space index fastest (like Isle)
                
        """
     
        n = x.shape[1]
        nx = int(n/self.Nt)
        nt = self.Nt
        for i in range(x.shape[0]):      # x is of shape = [n_samples,x_dim]
            x_i = x[i].detach().numpy()   
            if n == self.M_all[i].shape[0]: # this tests if the phi array has the right  dimensions
                for t in range(nt-1): # loop over time bloks
                    for x1 in range(nx): # loop over cords and kappa matrix
                        self.M_all[i][t*nx+x1][t*nx+x1] = 1.0  # diagonal term
                        for y in range(nx): # run over the kappa matrix
                            self.M_all[i][(t+1)*nx+x1][t*nx+y] = -self.expk[x1][y]*np.exp(1j*x_i[t*nx+y]) ## off-diagonal
                for x1 in range(nx): 
                    self.M_all[i][(nt-1)*nx+x1][(nt-1)*nx+x1] = 1.0  # diagonal term
                    for y in range(nx): # anti-periodic boundary condition
                        self.M_all[i][x1][(nt-1)*nx+y] = self.expk[x1][y]*np.exp(1j*x_i[(nt-1)*nx+y])
     
            else:
                print('# Error! phi and M have inconsistent dimensions!')
                return -1
        
        return 0
    
    
    # this routine calculates det(M[phi]M[-phi])
    def calcLogDetMM(self,x):
        """
        Function that calculates the determinant of Fermion matrices

        INPUT:
                x - phi field (array)

        OUTPUT:
                detMM - determinant of the matrices
                det_MM - array of determinant of matrices [x.shape[0] X 1]
        """
      
        self.Mphi(x) # update M with +1 phi
        det_MM = np.array([np.log(np.linalg.det(self.M_all[i])) for i in range(x.shape[0])])
#         detMM = np.log(np.linalg.det(M)) # calc detM with +1 phi

        self.Mphi(torch.negative(x)) # update M with -1 phi
        det_MM += np.array([np.log(np.linalg.det(self.M_all[i])) for i in range(x.shape[0])])
#         detMM += np.log(np.linalg.det(M)) # calc detM with -1 phi

        return det_MM
    
    def calcTrMM(self,x,sign):
        """
        Function that calculates the trace of a Fermion matrix

        INPUT:
                x - phi field (array)
                expk - exponential of hopping term (array)
                nt - number of time steps
                sign - sign of phi (+1 / -1)

        OUTPUT:
                TrMM - array of trace of the matrix , shape = [x.shape[0],x_dim]
        """
        TrMM = []
        n = x.shape[1]
        nx = int(n/self.Nt)
        nt = self.Nt
        self.Mphi(x) # update M
       
        invM = np.array([np.linalg.inv(self.M_all[i]) for i in range(x.shape[0])])  # only need to invert once!
        for i in range(x.shape[0]):
            x_i = x[i].detach().numpy()
            TrMM_i = [] # trace container
            for t in range(nt-1): # loop over time blocks
                for x1 in range(nx): # loop over sites  (space is fastest)
                    temp = 0 + 0j   #
                    for y in range(nx):
                        temp += invM[i][t*nx+x1][(t+1)*nx+y]*self.expk[y][x1]
                    TrMM_i.append(temp*(-sign*1j*np.exp(1j*x_i[t*nx+x1])))##
            for x1 in range(nx): # anti-periodic boundry conditions
                temp = 0 + 0j
                for y in range(nx):
                    temp += invM[i][(nt-1)*nx+x1][y]*self.expk[y][x1]
                TrMM_i.append(temp*sign*1j*np.exp(1j*x_i[(nt-1)*nx+x1]))##
            TrMM.append(np.array(TrMM_i))
       
        return np.array(TrMM)
    

    def _forward_step(self, x, v, step, aux=None):
        # transformation which corresponds for d=+1
        
        # pdb.set_trace()
        eps = safe_exp(self.alpha)
        t = self._format_time(step, tile=x.shape[0])  # time has size x.shape[0] x 2
        
        grad1 = self.grad_energy(x, aux=aux)  # gets gradient of sum of energy function at points x wrt x
       
        
        S1 = self.VNet([x, grad1, t, aux])  # this network is for momentum
             
        
        # here we get final outputs of our networks
        sv1 = 0.5 * eps * S1[0]  # Sv
        tv1 = S1[1]  # Tv
        fv1 = eps * S1[2]  # Qv

        v_h = v * safe_exp(sv1) - 0.5 * eps * ((safe_exp(fv1) * grad1) + tv1)  
   
        m, mb = self._get_mask(step)  # m and 1 - m
        # m, mb = self._gen_mask(x)
        
        
        X1 = self.XNet([v_h, m * x, t, aux])  # input is current momentum (output of the previous network v_h,
        # a half of current coordinates, time moment t)
        

        
        sx1 = (eps * X1[0])  # Sx
        tx1 = X1[1]  # Tx
        fx1 = eps * X1[2]  # Qx

        y = m * x + mb * (x * safe_exp(sx1) + eps * (safe_exp(fx1) * v_h + tx1))

        X2 = self.XNet([v_h, mb * y, t, aux])

        sx2 = (eps * X2[0])
        tx2 = X2[1]
        fx2 = eps * X2[2]

        x_o = mb * y + m * (y * safe_exp(sx2) + eps * (safe_exp(fx2) * v_h + tx2))

        grad2 = self.grad_energy(x_o, aux=aux)

        S2 = self.VNet([x_o, grad2, t, aux])  # last momentum update
        sv2 = (0.5 * eps * S2[0])
        tv2 = S2[1]
        fv2 = eps * S2[2]

        v_o = v_h * safe_exp(sv2) + 0.5 * eps * (-(safe_exp(fv2) * grad2) + tv2)
        log_jac_contrib = torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

        #  x_o - output of coordinates
        #  v_o - output of momentum
        #  log_jac_contrib - logarithm of Jacobian of forward transformation

        return x_o, v_o, log_jac_contrib

    def _backward_step(self, x_o, v_o, step, aux=None):
        # transformation which corresponds for d=-1
        eps = safe_exp(self.alpha)
        t = self._format_time(step, tile=x_o.shape[0])
        grad1 = self.grad_energy(x_o, aux=aux)

        S1 = self.VNet([x_o, grad1, t, aux])

        sv2 = (-0.5 * eps * S1[0])
        tv2 = S1[1]
        fv2 = eps * S1[2]

        v_h = safe_exp(sv2) * (v_o - 0.5 * eps * (-safe_exp(fv2) * grad1 + tv2))

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, mb * x_o, t, aux])

        sx2 = (-eps * X1[0])
        tx2 = X1[1]
        fx2 = eps * X1[2]

        y = mb * x_o + m * safe_exp(sx2) * (x_o - eps * (safe_exp(fx2) * v_h + tx2))

        X2 = self.XNet([v_h, m * y, t, aux])

        sx1 = (-eps * X2[0])
        tx1 = X2[1]
        fx1 = eps * X2[2]

        x = m * y + mb * safe_exp(sx1) * (y - eps * (safe_exp(fx1) * v_h + tx1))

        grad2 = self.grad_energy(x, aux=aux)
        S2 = self.VNet([x, grad2, t, aux])

        sv1 = (-0.5 * eps * S2[0])
        tv1 = S2[1]
        fv1 = eps * S2[2]

        v = safe_exp(sv1) * (v_h - 0.5 * eps * (-safe_exp(fv1) * grad2 + tv1))


        return x, v, torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)



    def hamiltonian(self, x, v, aux=None):
        H =  0.5 * torch.sum(v**2, dim=1) +  0.5 * torch.sum(x**2, dim=1)/(self.deltaU * self.Temp)
        H -= torch.real(torch.tensor(self.calcLogDetMM(x))) #shape = [n_samples,x_dim]
        return H    #if temperature divide potential energy by temp
        

    def grad_energy(self, x, aux=None):
        c = torch.real(torch.tensor(self.calcTrMM(x,1)))
        return torch.tensor(x/self.deltaU - 2*c,dtype=torchType) #shape = [n_samples,x_dim]

    def _gen_mask(self, x):
        b = np.zeros(self.x_dim)
        for i in range(self.x_dim):
            if i % 2 == 0:
                b[i] = 1
        b = b.astype('bool')
        nb = np.logical_not(b)

        return b.astype(numpyType), nb.astype(numpyType)

    def forward(self, x, init_v=None, aux=None, log_path=False, return_log_jac=False):
        # this function repeats _step_forward T times
#         print("n_samples=",self.n_samples)
        self.M_all= [self.M for i in range(self.n_samples)]  #[n_samples,[x_dim,x_dim]]  Fermion matrix
    
        if init_v is None:
            v = torch.normal(0.0,1.0,size=(self.n_samples,self.x_dim))
        else:
            v = init_v.clone()
       
        dN = x.shape[0]
        t = torch.tensor(0., dtype=torchType)
        j = torch.zeros(dN, dtype=torchType)

        x_init = x
        v_init = v

        while t < self.T:
            x, v, log_j = self._forward_step(x, v, t, aux=aux)
            j += log_j
            t += 1

        if return_log_jac:
            return x, v, j
       #shape x = [n_samples,x_dim]
       #shape v = [n_samples,x_dim]
       #shape p,j = [n_samples]

        return x, v, self.p_accept(x_init, v_init, x, v, j, aux=aux), j

    def backward(self, x, init_v=None, aux=None, return_log_jac=False):
        # this function repeats _step_backward T times
#         print("bn_samples=",self.n_samples)
        self.M_all= [self.M for i in range(self.n_samples)]  #[n_samples,[x_dim,x_dim]]  Fermion matrix
    
        if init_v is None:
            v = torch.normal(0.0,1.0,size=(self.n_samples,self.x_dim))
        else:
            v = init_v.clone()

        dN = x.shape[0]
        t = torch.tensor(0., dtype=torchType)
        j = torch.zeros(dN, dtype=torchType)

        x_init = x #.clone() #.data
        v_init = v #.clone() #.data

        while t < self.T:
            x, v, log_j = self._backward_step(x, v, self.T - t - 1, aux=aux)
            j += log_j
            t += 1

        if return_log_jac:
            return x, v, j

        return x, v, self.p_accept(x_init, v_init, x, v, j, aux=aux), j

    def p_accept(self, x0, v0, x1, v1, log_jac, aux=None):
        # metropolis accept/reject
        
        e_old = self.hamiltonian(x0, v0, aux=aux)
        e_new = self.hamiltonian(x1, v1, aux=aux)
        diff_H = e_old - e_new + log_jac
        other = torch.zeros_like(diff_H)
        p = torch.min(torch.exp(diff_H),torch.exp(other))
        
        return torch.where(torch.isfinite(p),p,torch.zeros_like(p))