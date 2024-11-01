import torch
from train.train_utils.loss_keys import LossKey

def init_mu_dict(loss_keys: list[LossKey], device)-> dict:
    mu_dict = {}
    for key in loss_keys:
            mu_dict[key.mu_key] = torch.tensor(1.0, device=device)
    return mu_dict

def init_aug_lagr_lambas(config: dict, loss_keys: list[LossKey])-> dict:
    n_points_config_dict = {
        'curv': 'surf_pts_nof_points',
        'div': 'surf_pts_nof_points',
        'eikonal': 'n_points_domain',
        'obst': 'n_points_obstacles',
        'if_normal': 'n_points_interfaces',
        'if': 'n_points_interfaces',
        'env': 'n_points_envelope',
        'scc': '',
        'dirichlet': '',
    }


    aug_lagr_lambda_dict = {}

    for key in loss_keys:
        if config['use_augmented_lagrangian']:
            if key.base_key == 'scc' or key.base_key=='dirichlet':
                n_points_config = 1
            elif key.base_key == 'if_normal':
                n_points_config = config[n_points_config_dict[key.base_key]] * config['nx'] * config['ginn_bsize']
            else:
                n_points_config = config[n_points_config_dict[key.base_key]] * config['ginn_bsize']
            
            #self.lambda_vec_dict[lamb] = torch.zeros(n_points_config, device=self.config['device'], requires_grad=True)
            aug_lagr_lambda_dict[key.lambda_key] = [torch.tensor(1.0, device=config['device']),
                                            None] #lambda vectors no longer used in augmented lagrangian, but kept for compatibility
        else:
            aug_lagr_lambda_dict[key.lambda_key] = None

    return aug_lagr_lambda_dict

def init_nu_dict(loss_keys: list[LossKey])-> dict:
    nu_dict = {}
    for key in loss_keys:
        nu_dict[key.nu_key] = 0.
    return nu_dict

def grad_norm_sub_losses(model, sub_loss_dict: dict)-> dict: 

    sub_grad_norm_dict = {}
    for loss_indent, loss_tensor in sub_loss_dict.items():
        if not torch.is_nonzero(loss_tensor):
            sub_grad_norm_dict[loss_indent.replace("loss_unweighted", "grad_norm")] = torch.tensor(0.0)
        elif loss_tensor.requires_grad:
            sub_grads = torch.autograd.grad(loss_tensor, model.parameters(), retain_graph=True, allow_unused=True)
            sub_grad_norm = torch.cat([grad.detach().flatten() for grad in sub_grads if grad is not None]).norm()
            sub_grad_norm_dict[loss_indent.replace("loss_unweighted", "grad_norm")] = sub_grad_norm
    return sub_grad_norm_dict

def lambda_balancing(lambda_dict: dict, sub_grad_norm_dict: dict, alpha: float)-> dict:
    sub_grad_sum = sum(sub_grad_norm_dict.values())
    for lambda_key, lambda_value in lambda_dict.items():
        sub_grad_norm_key = lambda_key.replace('lambda', 'grad_norm')
        sub_grad_norm = sub_grad_norm_dict[sub_grad_norm_key]
        if not torch.is_nonzero(sub_grad_norm):
            continue
        lambda_value_update = sub_grad_sum / sub_grad_norm

        lambda_dict[lambda_key] = alpha * lambda_value + (1 - alpha) * lambda_value_update

    return lambda_dict

def lambda_vec_balancing(lambda_vec_dict: dict, sub_loss_dict: dict, eta: float)-> dict:
    pass 

def adaptive_penalty_update(loss_keys: list[LossKey], mu_dict: dict, aug_lagr_lambda_dict: dict, nu_dict: dict, sub_loss_unweighted_dict: dict, config: dict)-> dict:
    gamma = 0.01
    alpha = 0.9
    epsilon = 1.e-08

    with torch.no_grad():
        for key in loss_keys:
            nu_dict[key.nu_key] = nu_dict[key.nu_key] * alpha + (1-alpha) * sub_loss_unweighted_dict[key.loss_unweighted_key]
            mu_dict[key.mu_key] = gamma / (torch.sqrt(nu_dict[key.nu_key]) + epsilon)
            aug_lagr_lambda_dict[key.lambda_key][0] = aug_lagr_lambda_dict[key.lambda_key][0] + \
                                                      mu_dict[key.mu_key] * torch.sqrt(sub_loss_unweighted_dict[key.loss_unweighted_key])

    return mu_dict, aug_lagr_lambda_dict, nu_dict



def scale_losses(lambda_dict: dict, sub_loss_dict: dict)-> dict:
    for lambda_key, lambda_value in lambda_dict.items():
        sub_loss_key = 'loss' + lambda_key.replace('lambda', '')
        sub_loss_dict[sub_loss_key] = lambda_value * sub_loss_dict[sub_loss_key]

    return sub_loss_dict