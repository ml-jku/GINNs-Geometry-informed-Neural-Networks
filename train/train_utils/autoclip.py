import numpy as np
import torch

from utils import set_and_true

class AutoClip:
    def __init__(self, config):
        """
        Initializes the AutoClip instance, inspired by https://arxiv.org/abs/2007.14469

        :param history_size: Number of gradient norms to keep in history.
        :param percentile: Percentile value to determine the clip threshold.
        :param default_clip_value: Default clip value to use if history is too short or AutoClip is disabled.
                                   Set to np.inf to completely disable clipping.
        :param min_history_length: Minimum length of history to use percentile-based clipping.
        :param enabled: Flag to enable or disable adaptive clipping.
        """
        
        self.grad_clip_enabled = set_and_true('grad_clipping_on', config)
        self.auto_clip_enabled = set_and_true('auto_clip_on', config)
        self.default_clip_value = config.get('grad_clip', None)
        self.percentile = config.get('auto_clip_percentile', None)

        history_size=config.get('auto_clip_hist_len', None)
        min_history_length=config.get('auto_clip_min_len', None)
        
        self.min_history_length = min_history_length
        
        if not (self.grad_clip_enabled and self.auto_clip_enabled):
            self.history_size = 1
        else:        
            self.history_size = history_size
        
        self.gradient_norms = torch.zeros(self.history_size)
        self.cur_idx = 0
        self.cnt = 0


    def update_gradient_norm_history(self, gradient_norm):
        """
        Updates the history of gradient norms.
        :param gradient_norm: The norm of the most recent gradient.
        """
        self.gradient_norms[self.cur_idx] = gradient_norm
        self.cur_idx = (self.cur_idx + 1) % self.history_size
        self.cnt += 1

    def get_clip_value(self):
        """
        Calculates and returns the current gradient clipping value.
        :return: Gradient clipping threshold based on the percentile, default value, or np.inf if clipping is disabled.
        """
        if not self.grad_clip_enabled:
            return np.inf
        
        if not self.auto_clip_enabled or self.cnt < self.min_history_length:
            return self.default_clip_value  # Return default value if AutoClip is disabled or history is too short
        
        return torch.quantile(self.gradient_norms, self.percentile)
    
    def get_last_gradient_norm(self):
        """
        Returns the norm of the last gradient.
        :return: The norm of the last gradient.
        """
        if self.grad_clip_enabled:
            return self.gradient_norms[self.cur_idx - 1]
        else:
            # return nan
            return torch.tensor(float('nan'))            

    def grad_norm(self, parameters):
        """
        Computes the norm of the gradients of the parameters.
        Args:
            parameters (iterable): An iterable of torch.Tensor containing the parameters of the model.
        """        
        grads = [param.grad.detach().flatten() for param in parameters if param.grad is not None ]
        grad_norm = torch.cat(grads).norm()
        return grad_norm