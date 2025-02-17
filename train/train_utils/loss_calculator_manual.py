import torch
from train.train_utils.loss_calculator_base import LossCalculator


class ManualWeightedLoss(LossCalculator):
    def __init__(self, scalar_loss_keys, lambda_dict, field_loss_keys=[], **kwargs):
        super().__init__()
        self.loss_keys = scalar_loss_keys + field_loss_keys
        self.scalar_loss_keys = scalar_loss_keys
        self.field_loss_keys = field_loss_keys
        
        self.lamda = { key.lambda_key: torch.tensor(lambda_dict[key], device=torch.get_default_device()) for key in self.loss_keys }
        
        self.loss_weighted_dict = {}
        self.loss_unweighted_dict = {}
        
    def _compute_scalar_subloss(self, key, sub_loss):
        sub_loss_unweighted = torch.sqrt(sub_loss)
        sub_loss_weighted = self.lamda[key.lambda_key] * sub_loss_unweighted
        
        # update log dicts
        self.loss_unweighted_dict[key.loss_unweighted_key] = sub_loss_unweighted
        self.loss_weighted_dict[key.loss_key] = sub_loss_weighted
        
        return sub_loss_weighted
    
    def _compute_field_subloss(self, key, sub_loss, grad_field):
            
            # sub_loss_unweighted = torch.sqrt(torch.mean(sub_loss))
            sub_loss_unweighted = torch.mean(sub_loss)
            sub_loss_weighted = self.lamda[key.lambda_key] * sub_loss_unweighted
    
            ### Fill dicts needed for wandb logging 
            self.loss_unweighted_dict[key.loss_unweighted_key] = sub_loss_unweighted
            self.loss_weighted_dict[key.loss_key] = sub_loss_weighted
    
            ### Fill dict which is acumulated for the backward pass
            # self.sub_grad_field_dict[key.loss_key] = self.batch_grad_field_sqrt_derivative(x_value_batch=sub_loss, x_field_grad_batch=grad_field, scalar_batch=self.lamda[key.lambda_key].unsqueeze(0).expand(sub_loss.shape[0], 1))
            self.sub_grad_field_dict[key.loss_key] = grad_field
            return sub_loss_weighted
        
    def compute_loss_and_save_sublosses(self, losses_dict):
        assert set(losses_dict.keys()) == set(self.loss_keys)
        
        loss = 0.
        for key, val in losses_dict.items():
            if key in self.field_loss_keys:
                # for a field, val is a tuple (batch_C, batch_dCdrho)
                loss += self._compute_field_subloss(key, *val)
            elif key in self.scalar_loss_keys:
                loss += self._compute_scalar_subloss(key, val)
            else:
                raise ValueError(f"Key {key} not found in scalar or field loss keys")
            
        return loss
    
    def adaptive_update(self):
        # no adaptive update needed
        pass
    
    def get_dicts(self):
        return [self.loss_weighted_dict, self.loss_unweighted_dict, self.lamda]
                