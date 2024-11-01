import time
from torch.optim import Adam, LBFGS, Optimizer
from .gd import GD

class Adam_LBFGS_GD(Optimizer):
    def __init__(self, params, switch_epoch1, switch_epoch2, adam_params, lbfgs_params, gd_params):

        self.switch_epoch1 = switch_epoch1
        self.switch_epoch2 = switch_epoch2
        self.params = list(params)
        self.adam = Adam(self.params, **adam_params)
        self.lbfgs = LBFGS(self.params, **lbfgs_params)
        self.gd = GD(self.params, **gd_params)

        super(Adam_LBFGS_GD, self).__init__(self.params, defaults={})

        self.state['epoch'] = 0

    def step(self, closure=None):
        if self.state['epoch'] < self.switch_epoch1:
            self.adam.step(closure)
            self.state['epoch'] += 1

        elif self.state['epoch'] < self.switch_epoch2:
            if self.state['epoch'] == self.switch_epoch1:
                print(f'Switching to LBFGS optimizer at epoch {self.state["epoch"]} at time {time.time()}')
            self.lbfgs.step(closure)
            self.state['epoch'] += 1
            
        else:
            if self.state['epoch'] == self.switch_epoch2:
                print(f'Switching to GD optimizer at epoch {self.state["epoch"]} at time {time.time()}')
            _, grad = self.gd.step(closure)
            self.state['epoch'] += 1
            return grad

        