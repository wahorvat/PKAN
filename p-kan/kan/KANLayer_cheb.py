import torch
import torch.nn as nn
import numpy as np
from .cheb import cheb_basis, coef2curve_cheb, curve2coef_cheb
from .utils import sparse_mask

class ChebKANLayer(nn.Module):
    """
    KANLayer class using Chebyshev polynomials instead of splines
    """
    def __init__(self, in_dim=3, out_dim=2, degree=5, noise_scale=0.5, 
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, 
                 base_fun=torch.nn.SiLU(), device='cpu', sparse_init=False):
        super(ChebKANLayer, self).__init__()
        
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.degree = degree

        # Initialize Chebyshev coefficients with noise
        noises = (torch.rand(self.in_dim, self.out_dim, self.degree + 1) - 1/2) * noise_scale
        self.coef = torch.nn.Parameter(noises)
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        # Initialize scaling parameters
        self.scale_base = torch.nn.Parameter(
            scale_base_mu * 1 / np.sqrt(in_dim) + 
            scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)
        )
        
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * self.mask)
        self.base_fun = base_fun
        
        self.to(device)
        
    def to(self, device):
        super(ChebKANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        '''
        ChebKANLayer forward given input x
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations
            postacts : 3D torch.float
                outputs of activation functions
            postchebyshev : 3D torch.float
                outputs of Chebyshev functions
        '''
        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        base = self.base_fun(x) # (batch, in_dim)
        y = coef2curve_cheb(x, self.coef, self.degree)
        
        postchebyshev = y.clone().permute(0,2,1)
            
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
            
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postchebyshev

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            cheb : ChebKANLayer
        '''
        cheb = ChebKANLayer(len(in_id), len(out_id), self.degree, base_fun=self.base_fun)
        cheb.coef.data = self.coef[in_id][:,out_id]
        cheb.scale_base.data = self.scale_base[in_id][:,out_id]
        cheb.scale_sp.data = self.scale_sp[in_id][:,out_id]
        cheb.mask.data = self.mask[in_id][:,out_id]

        cheb.in_dim = len(in_id)
        cheb.out_dim = len(out_id)
        return cheb
    
    def swap(self, i1, i2, mode='in'):
        '''
        swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out') 
        
        Args:
        -----
            i1 : int
            i2 : int
            mode : str
                mode = 'in' or 'out'
            
        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=2, out_dim=2, num=5, k=3)
        >>> print(model.coef)
        >>> model.swap(0,1,mode='in')
        >>> print(model.coef)
        '''
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:,i1], data[:,i2] = data[:,i2].clone(), data[:,i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)
