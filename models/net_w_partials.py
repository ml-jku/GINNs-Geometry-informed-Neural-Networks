

from turtle import st
from typing import Any


class NetWithPartials:
    '''
    The only stateful part of the model is the parameters.
    '''
    
    def __init__(self, f_, vf_x_, vf_xx_, params_) -> None:
        self.f_ = f_
        self.vf_x_ = vf_x_
        self.vf_xx_ = vf_xx_
        self.params_ = params_
        
    def f(self, *args: Any) -> Any:
        return self.f_(self.params_, *args)
    
    def vf_x(self, *args: Any) -> Any:
        return self.vf_x_(self.params_, *args)
    
    def vf_xx(self, *args: Any) -> Any:
        return self.vf_xx_(self.params_, *args)
    
    def detach(self) -> None:
        ## Make the parameters not request gradient computation by creating a detached copy.
        ## The copy is shallow, meaning it shares memory, and is tied to parameter updates in the outer loop.
        new_params = {key: tensor.detach() for key, tensor in self.params_.items()}
        new_net = NetWithPartials(self.f_, self.vf_x_, self.vf_xx_, new_params)
        return new_net