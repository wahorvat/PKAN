import torch

def cheb_basis(x, degree=0, device='cpu'):
    '''
    Evaluate x on Chebyshev polynomial basis functions
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (batch_size, in_dim)
        degree : int
            maximum degree of Chebyshev polynomials
        device : str
            device
    
    Returns:
    --------
        polynomial values : 3D torch.tensor
            shape (batch_size, in_dim, degree+1)
    
    Example
    -------
    >>> x = torch.rand(100,2)
    >>> cheb_basis(x, degree=3).shape
    '''
    # Normalize x to [-1, 1] using tanh
    x = torch.tanh(x)
    
    # Reshape for broadcasting
    x = x.unsqueeze(dim=2).expand(-1, -1, degree + 1)
    
    # Generate angles
    angles = x.acos()
    degrees = torch.arange(0, degree + 1, device=x.device)
    
    # Compute Chebyshev polynomials using cosine formula
    # Tn(x) = cos(n * arccos(x))
    values = torch.cos(angles * degrees)
    
    return values

def coef2curve_cheb(x_eval, coef, degree, device="cpu"):
    '''
    Converting Chebyshev coefficients to curves by evaluating the polynomial.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch_size, in_dim)
        coef : 3D torch.tensor
            shape (in_dim, out_dim, degree+1)
        degree : int
            maximum degree of Chebyshev polynomials
        device : str
            device
    
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch_size, in_dim, out_dim)
    '''
    # Get Chebyshev basis values
    basis_values = cheb_basis(x_eval, degree=degree)
    
    # Compute the weighted sum using einsum
    y_eval = torch.einsum('ijk,jlk->ijl', basis_values, coef.to(basis_values.device))
    
    return y_eval

def curve2coef_cheb(x_eval, y_eval, degree, lamb=1e-8):
    '''
    Converting curves to Chebyshev coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch_size, in_dim)
        y_eval : 3D torch.tensor
            shape (batch_size, in_dim, out_dim)
        degree : int
            maximum degree of Chebyshev polynomials
        lamb : float
            regularization parameter for least squares
    
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, degree+1)
    '''
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    
    # Get Chebyshev basis matrix
    mat = cheb_basis(x_eval, degree=degree)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, degree + 1)
    
    # Reshape y_eval for least squares
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
    device = mat.device
    
    # Solve least squares problem with regularization
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
    
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:,:,:,0]
    
    return coef

def normalize_domain(x, input_range=[-1, 1]):
    '''
    Normalize input domain to [-1, 1] for Chebyshev polynomials
    
    Args:
    -----
        x : torch.tensor
            input values
        input_range : list
            [min, max] of input domain
    
    Returns:
    --------
        normalized_x : torch.tensor
            input normalized to [-1, 1]
    '''
    min_val, max_val = input_range
    return 2 * (x - min_val) / (max_val - min_val) - 1