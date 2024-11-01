from math import log
import torch
from torch.optim import Adam, LBFGS, AdamW
from train.opt.adam_lbfgs import Adam_LBFGS
from train.opt.adam_lbfgs_nncg import Adam_LBFGS_NNCG
from train.opt.adam_lbgfs_gd import Adam_LBFGS_GD
from train.train_utils.autoclip import AutoClip

# code adapted from: https://anonymous.4open.science/r/opt_for_pinns-9246/src/train_utils.py

def opt_step(opt, epoch, model, loss_fn, z, z_corners, batch, auto_clip: AutoClip):
    
    log_dict = {}
    
    # Update the preconditioner for NysNewtonCG
    if isinstance(opt, Adam_LBFGS_NNCG) and epoch >= opt.switch_epoch2 and epoch % opt.precond_update_freq == 0:
        print(f'Updating preconditioner at epoch {epoch}')
        opt.zero_grad()
        loss, _, __, ___ = loss_fn(z, epoch, batch, z_corners)
        grad_tuple = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        opt.nncg.update_preconditioner(grad_tuple)
            
    # second optimizer
    if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and epoch >= opt.switch_epoch2: 
        #print(f'Using default closure at epoch {epoch}')
        def closure():
            opt.zero_grad()
            loss, sub_loss_dict, sub_loss_unweighted_dict, sub_al_loss_dict, al_vec_l2_dict = loss_fn(z, epoch, batch, z_corners)
            grad_tuple = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            if 'loss' not in log_dict:
                log_dict['loss'] = loss
                log_dict.update(sub_loss_dict)
                log_dict.update(sub_loss_unweighted_dict)
                log_dict.update(sub_al_loss_dict)
                log_dict.update(al_vec_l2_dict)
            return loss, grad_tuple
    else:
        def closure():
            opt.zero_grad()
            loss, sub_loss_dict, sub_loss_unweighted_dict, sub_al_loss_dict, al_vec_l2_dict = loss_fn(z, epoch, batch, z_corners)
            loss.backward()
            if 'loss' not in log_dict:
                log_dict.update(sub_loss_dict)
                log_dict.update(sub_loss_unweighted_dict)
                log_dict.update(sub_al_loss_dict)
                log_dict.update(al_vec_l2_dict)
                log_dict['loss'] = loss
                grad_norm = auto_clip.grad_norm(model.parameters())
                if torch.isnan(grad_norm).any():
                    pass
                log_dict['grad_norm_pre_clip'] = grad_norm
                if auto_clip.grad_clip_enabled:
                    auto_clip.update_gradient_norm_history(grad_norm)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), auto_clip.get_clip_value())
 
            return loss

    if isinstance(opt, (Adam_LBFGS_NNCG, Adam_LBFGS_GD)) and epoch >= opt.switch_epoch2:
        grad = opt.step(closure)
    else:
        opt.step(closure)
        
    return log_dict


def get_opt(opt_name, opt_params, model_params, lambda_vec_dict):
    if opt_name == 'adam':
        param_list = [{'params': model_params}]
        for key, value in lambda_vec_dict.items():
            pass
            #if value is not None:
            #    if key != 'lambda_scc':
            #        param_list.append({'params': value[0], 'maximize': True})
        return Adam(param_list) #, **opt_params)  ## use default parameters
    elif opt_name == 'lbfgs':
        if "history_size" in opt_params:
            opt_params["history_size"] = int(opt_params["history_size"])
        return LBFGS(model_params, **opt_params, line_search_fn='strong_wolfe')
    elif opt_name == 'adam_lbfgs':
        if "switch_epochs" not in opt_params:
            raise KeyError("switch_epochs is not specified for Adam_LBFGS optimizer.")
        switch_epochs = opt_params["switch_epochs"]

        # Ensure switch_epochs is a list of integers
        if not isinstance(switch_epochs, list):
            switch_epochs = [switch_epochs]
        switch_epochs = [int(epoch) for epoch in switch_epochs]

        # Get parameters for Adam and LBFGS, remove the prefix "adam_" and "lbfgs_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        
        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS(model_params, switch_epochs, adam_params, lbfgs_params)
    elif opt_name == 'adam_lbfgs_nncg':
        if "switch_epoch_lbfgs" not in opt_params:
            raise KeyError("switch_epoch_lbfgs is not specified for Adam_LBFGS_NNCG optimizer.")
        if "switch_epoch_nncg" not in opt_params:
            raise KeyError("switch_epoch_nncg is not specified for Adam_LBFGS_NNCG optimizer.")
        if "precond_update_freq" not in opt_params:
            raise KeyError("precond_update_freq is not specified for Adam_LBFGS_NNCG optimizer.")
        switch_epoch_lbfgs = int(opt_params["switch_epoch_lbfgs"])
        switch_epoch_nncg = int(opt_params["switch_epoch_nncg"])
        precond_update_freq = int(opt_params["precond_update_freq"])

        # Get parameters for Adam, LBFGS, and NNCG, remove the prefix "adam_", "lbfgs_", and "nncg_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        nncg_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("nncg_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        nncg_params["line_search_fn"] = "armijo"

        nncg_params["verbose"] = True

        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])
        if "rank" in nncg_params:
            nncg_params["rank"] = int(nncg_params["rank"])

        return Adam_LBFGS_NNCG(model_params, switch_epoch_lbfgs, switch_epoch_nncg, precond_update_freq, adam_params, lbfgs_params, nncg_params)
    elif opt_name == 'adam_lbfgs_gd':
        if "switch_epoch_lbfgs" not in opt_params:
            raise KeyError("switch_epoch_lbfgs is not specified for Adam_LBFGS_GD optimizer.")
        if "switch_epoch_gd" not in opt_params:
            raise KeyError("switch_epoch_gd is not specified for Adam_LBFGS_GD optimizer.")
        switch_epoch_lbfgs = int(opt_params["switch_epoch_lbfgs"])
        switch_epoch_gd = int(opt_params["switch_epoch_gd"])

        # Get parameters for Adam, LBFGS, and GD, remove the prefix "adam_", "lbfgs_", and "gd_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        gd_params = {k[3:]: v for k, v in opt_params.items() if k.startswith("gd_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"
        gd_params["line_search_fn"] = "armijo"

        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS_GD(model_params, switch_epoch_lbfgs, switch_epoch_gd, adam_params, lbfgs_params, gd_params)
    else:
        raise ValueError(f'Optimizer {opt_name} not supported')