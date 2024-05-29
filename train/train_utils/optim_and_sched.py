from asyncio import start_server
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class FixedStepOptimizer(Optimizer):
    def __init__(self, input_list, step_size, eps=1e-15):
        """
        Args:
            input_tensor (torch.Tensor): The input tensor to be optimized.
            step_size (float): The fixed step size for the update.
        """
        self.step_size = step_size
        self.eps = eps
        defaults = dict(step_size=step_size)
        super(FixedStepOptimizer, self).__init__(input_list, defaults)

    def step(self): 
        for group in self.param_groups: 
            for p in group['params']: 
                grad_norm = torch.linalg.vector_norm(p.grad.data, dim=-1, ord=2) + self.eps
                p.data -= group['step_size'] * p.grad.data / grad_norm.unsqueeze(-1)
                # TODO: maybe plot histogram of grad_norms to adjust eps parameter (eps is important for fast convergence)
                # print(f'grad_norm.mean: {grad_norm.mean()}') 


class LinearScheduler():
    
    def __init__(self, decay_steps, start=1.0, stop=0.0) -> None:
        self.decay_steps=decay_steps
        self.start=start
        self.stop=stop

    def get_val(self, step: int):
        if step > self.decay_steps:
            return self.stop
        
        return self.stop + (self.start - self.stop) * ((self.decay_steps - step) / self.decay_steps)
    
class StepScheduler():
    
    def __init__(self, decay_steps, start=1.0, stop=0.0) -> None:
        self.decay_steps=decay_steps
        self.start=start
        self.stop=stop

    def get_val(self, step: int):
        if step > self.decay_steps:
            return self.stop
        else:
            return self.start
        
        

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a linear learning rate scheduler with warmup.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of warmup steps.
        num_training_steps (int): The total number of training steps.
        last_epoch (int): The index of the last epoch.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler with warmup.
    """
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / num_warmup_steps
        return max(1.0e-9, float(num_training_steps - current_step) / num_training_steps - num_warmup_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch)