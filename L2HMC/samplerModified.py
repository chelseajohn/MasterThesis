import torch
import torch.nn as nn
import numpy as np
import pdb

torchType = torch.float32


def propose(x,n_samples, dynamics, init_v=None, aux=None, do_mh_step=False, return_log_jac=False, temperature=None):
    dynamics.n_samples = n_samples
    if dynamics.hmc:
        Lx, Lv, px, _ = dynamics.forward(x, init_v=init_v, aux=aux)
        return Lx, Lv, px, [tf_accept(x, Lx,px)]  # new coordinates, new momenta, new acceptance probability,[accepted coordinate,mask]
    
    else:
       
        if temperature is not None:
            dynamics.temperature = temperature
    
        # sample mask for forward/backward
        mask = (2 * torch.tensor(np.random.rand(x.shape[0],1))).type(torch.int32).type(torchType)
        Lx1, Lv1, px1, log_jac_f = dynamics.forward(x.clone(), aux=aux, return_log_jac=return_log_jac) 
        Lx2, Lv2, px2,log_jac_b = dynamics.backward(x.clone(), aux=aux, return_log_jac=return_log_jac)
 
    Lx = mask * Lx1 + (1 - mask) * Lx2  # by this we imitate the random choice of d (direction)
    Lv = None
    if init_v is not None:
        Lv = mask * Lv1 + (1 - mask) * Lv2
        
    log_jac = torch.squeeze(mask, dim=1) * log_jac_f + torch.squeeze((1 - mask), dim=1) * log_jac_b
    px = torch.squeeze(mask, dim=1) * px1 + torch.squeeze(1 - mask, dim=1) * px2 
    
         
   
    outputs = []
    directions = None
    if do_mh_step:
        new_Lx, directions = tf_accept(x=x, Lx=Lx, px=px)
        outputs.append(new_Lx)
        
    return  Lx, Lv, px, outputs, log_jac, directions  # new coordinates, new momenta, new acceptance probability, outputs for accpeted coodinates, log_jac,taking MH acceptance in account

def tf_accept(x, Lx, px):
  
    probs = torch.tensor(np.random.rand(*list(px.shape)), dtype=torchType)
    mask = (probs <= px)[:, None] 

    return torch.where(mask, Lx, x).detach(), mask.squeeze()

