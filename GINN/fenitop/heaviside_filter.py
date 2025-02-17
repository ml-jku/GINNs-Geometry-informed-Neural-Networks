import torch

def heaviside(rho, beta, nf_is_density, theta=0.5):
    '''
    Heaviside filter function
    Theta is set to 0.5 in
    - Filters in topology optimization based on Helmholtz-type differential equations B. S. Lazarov∗,† and O. Sigmund
    - FEniTop: a simple FEniCSx implementation for 2D and 3D topology optimization supporting parallel computing
    '''
    
    if not nf_is_density:
        rho = torch.sigmoid(-rho)
    
    if type(beta) in [int, float]:
        beta = torch.tensor(beta, dtype=rho.dtype, device=rho.device)
    # Compute the components of the equation
    tanh_beta_theta = torch.tanh(beta * theta)
    tanh_beta_rho_minus_theta = torch.tanh(beta * (rho - theta))
    tanh_beta_one_minus_theta = torch.tanh(beta * (1 - theta))

    # Compute the numerator and the denominator
    numerator = tanh_beta_theta + tanh_beta_rho_minus_theta
    denominator = tanh_beta_theta + tanh_beta_one_minus_theta

    # Compute rho_bar
    rho_bar = numerator / denominator

    return rho_bar
