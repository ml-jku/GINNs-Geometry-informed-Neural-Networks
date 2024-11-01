import torch
from torch.optim import Optimizer
from functools import reduce

def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * gx.dot(dx):
        t *= beta
        f1 = f(x, t, dx)
    return t

class GD(Optimizer):
    def __init__(self, params, lr=1.0, line_search_fn=None, verbose=False):
        defaults = dict(lr=lr, line_search_fn=line_search_fn)
        self.line_search_fn = line_search_fn
        self.verbose = verbose
        self.n_iters = 0
        super(GD, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError(
                "GD doesn't currently support per-parameter options (parameter groups)")

        if self.line_search_fn is not None and self.line_search_fn != 'armijo':
            raise ValueError("GD only supports Armijo line search")

        self._params = self.param_groups[0]['params']
        self._params_list = list(self._params)
        self._numel_cache = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        g = torch.cat([grad.view(-1)
                      for grad in grad_tuple if grad is not None])

        # one step update
        for group_idx, group in enumerate(self.param_groups):
            if self.line_search_fn == 'armijo':
                x_init = self._clone_param()

                def obj_func(x, t, dx):
                    self._add_grad(t, dx)
                    loss = float(closure()[0])
                    self._set_param(x)
                    return loss

                # Use -d for convention
                t = _armijo(obj_func, x_init, g, -g, group['lr'])
            else:
                t = group['lr']

            self.state[group_idx]['t'] = t

            # update parameters
            ls = 0
            for p in group['params']:
                np = torch.numel(p)
                dp = g[ls:ls+np].view(p.shape)
                ls += np
                # if p.grad is None:
                #     continue
                p.data.add_(-dp, alpha=t)

        self.n_iters += 1

        return loss, g
    
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data
