from re import sub
import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys
import math
from typing import get_type_hints
from types import FunctionType

from models.point_wrapper import PointWrapper
from train.train_utils.loss_calculator_base import LossCalculator


class AdaptiveAugmentedLagrangianLoss(LossCalculator):
    def __init__(self, scalar_loss_keys, obj_key, lambda_dict, alpha, gamma, epsilon, field_loss_keys=[], **kwargs):
        super().__init__()

        self.loss_keys = scalar_loss_keys + field_loss_keys
        self.scalar_loss_keys = scalar_loss_keys
        self.field_loss_keys = field_loss_keys
        self.obj_key = obj_key
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        subloss_keys = [key for key in self.loss_keys if key != obj_key]
        
        # print(f'WARNING: In contrast to previous implementation, the lambda_dict is not set to all 1s, but to the config lambda values')
        self.lamda = { key.lambda_key: torch.tensor(lambda_dict[key], device=torch.get_default_device()) for key in self.loss_keys }
        # self.lamda = { key.lambda_key: torch.tensor(1.0, device=torch.get_default_device()) for key in subloss_keys }
        # # add objective lambda explicitly and don't set to 1, as it is like a constant scale factor of the loss
        # self.lamda[obj_key.lambda_key] = torch.tensor(lambda_dict[obj_key], device=torch.get_default_device())
        self.mu = { key.mu_key: torch.tensor(1.0, device=torch.get_default_device()) for key in subloss_keys }
        self.nu = { key.nu_key: torch.tensor(0.0, device=torch.get_default_device()) for key in subloss_keys }

        self.loss_unweighted_dict = {}
        self.lagrangian_loss_dict = {}
        self.mu_loss_dict = {}
        
    def _compute_scalar_subloss(self, key, sub_loss, is_objective=False):
        
        sub_loss_unweighted = torch.sqrt(sub_loss)
        # from run jiwtocmu, sqrt seems to be important
        # print(f'WARNING: testing if we should take sqrt of the loss or not')
        # sub_loss_unweighted = sub_loss
        lagrangian_loss = self.lamda[key.lambda_key] * sub_loss_unweighted
        # L = lambda_o * objective + lambda_i * C_i + mu_i * sqrt(C).square()
        # for the objective we only want the lambda_o * objective  
        if is_objective:
            mu_loss = 0.    
        else:
            mu_loss = 0.5 * self.mu[key.mu_key] * sub_loss_unweighted
        sub_loss_weighted = lagrangian_loss + mu_loss
        
        # update log dicts
        self.loss_unweighted_dict[key.loss_unweighted_key] = sub_loss_unweighted
        self.lagrangian_loss_dict[key.lagrangian_key] = lagrangian_loss
        self.mu_loss_dict[key.mu_key] = mu_loss
        self.loss_weighted_dict[key.loss_key] = sub_loss_weighted
                
        return sub_loss_weighted
    
    def _compute_field_subloss(self, key, sub_loss, grad_field, is_objective):

        # TODO: AR IMPORTANT: double-check if torch.mean or if we should take sqrt first
        # sub_loss_unweighted = torch.sqrt(sub_loss)
        sub_loss_unweighted = sub_loss
        lambda_loss = self.lamda[key.lambda_key] * sub_loss_unweighted
        if is_objective:
            mu_loss = 0.    
        else:
            mu_loss = 0.5 * self.mu[key.mu_key] * sub_loss_unweighted
        sub_loss_weighted = lambda_loss + mu_loss

        ### Fill dicts needed for wandb logging 
        self.loss_unweighted_dict[key.loss_unweighted_key] = sub_loss_unweighted
        self.lagrangian_loss_dict[key.lambda_key] = lambda_loss
        self.mu_loss_dict[key.mu_key] = mu_loss
        self.loss_weighted_dict[key.loss_key] = sub_loss_weighted

        # Note: L = lambda_o * objective + lambda_i * C_i + mu_i * sqrt(C).square()
        # for the objective we only want the lambda_o * objective        
        
        # print(f'WARNING A->E: for field losses we do not take the sqrt of the loss, but the loss itself (also see loss above)')        
        ### Fill dict which is acumulated for the backward pass
        lambda_field = self.batch_grad_field_linear_derivative(x_value_batch=sub_loss_unweighted, x_field_grad_batch=grad_field, 
                                                             scalar_batch=self.lamda[key.lambda_key])
        if is_objective:
            mu_field = 0.
        else:
            mu_field = self.batch_grad_field_square_derivative(x_value_batch=sub_loss_unweighted, x_field_grad_batch=grad_field, 
                                                               scalar_batch=self.mu[key.mu_key])
        
        self.sub_grad_field_dict[key] = mu_field + lambda_field
        return sub_loss_weighted
    
    def compute_loss_and_save_sublosses(self, losses_dict):
        assert set(losses_dict.keys()) == set(self.loss_keys), f"Keys in losses_dict {losses_dict.keys()} do not match loss_keys {self.loss_keys}"
        
        loss = 0.
        for key, val in losses_dict.items():
            is_objective = key==self.obj_key
            if key in self.field_loss_keys:
                assert val[0].dim() == 0, f"Field loss {key} should be a scalar"
                assert val[0] >= 0, f"Field loss {key} should be non-negative"
                # for a field, val is a tuple, e.g. (batch_C, batch_dCdrho)
                loss += self._compute_field_subloss(key, *val, is_objective=is_objective)  # unpack tuple
            elif key in self.scalar_loss_keys:
                assert val >= 0, f"Scalar loss {key} should be non-negative"
                loss += self._compute_scalar_subloss(key, val, is_objective=is_objective)
            else:
                raise ValueError(f"Key {key} not found in scalar or field loss keys")
        return loss
    
    def adaptive_update(self):
        with torch.no_grad():
            for key in self.loss_keys:
                if key == self.obj_key:
                    continue
                
                self.nu[key.nu_key] = self.nu[key.nu_key] * self.alpha + (1-self.alpha) * self.loss_unweighted_dict[key.loss_unweighted_key]
                if self.nu[key.nu_key] == 0:
                    # NOTE: The loss has never been > 0, so we keep mu and lambda at the initial values
                    continue
                
                self.mu[key.mu_key] = self.gamma / (torch.sqrt(self.nu[key.nu_key]) + self.epsilon)

                self.lamda[key.lambda_key] = self.lamda[key.lambda_key] + \
                                             self.mu[key.mu_key] * torch.sqrt(self.loss_unweighted_dict[key.loss_unweighted_key])

    def get_dicts(self):
        return [self.loss_weighted_dict, self.loss_unweighted_dict, self.lagrangian_loss_dict, self.lamda, self.mu]
