from typing import Any
import torch


class NetWithPartials:
    '''
    The only stateful part of the model is the parameters.
    '''
    
    def __init__(self, f_, f_x_, vf_x_, f_xx_, vf_xx_, params_, vf_z_=None) -> None:
        self.f_ = f_
        self.vf_x_ = vf_x_
        self.f_x_ = f_x_
        self.f_xx_ = f_xx_
        self.vf_xx_ = vf_xx_
        self.params_ = params_
        self.vf_z_ = vf_z_
        
    def f(self, *args: Any) -> Any:
        return self.f_(self.params_, *args)
    
    def f_x(self, *args: Any) -> Any:
        return self.f_x_(self.params_, *args)
    
    def vf_x(self, *args: Any) -> Any:
        return self.vf_x_(self.params_, *args)

    def f_xx(self, *args: Any) -> Any:
        return self.f_xx_(self.params_, *args)
    
    def vf_xx(self, *args: Any) -> Any:
        return self.vf_xx_(self.params_, *args)
    
    def vf_z(self, *args: Any) -> Any:
        if self.vf_z_ is None:
            raise ValueError('vf_z not defined. Have you specified nz > 0?')
        return self.vf_z_(self.params_, *args)
    
    def vf_theta(self, *args: Any) -> Any:
        if self.vf_theta_ is None:
            raise ValueError('vf_theta_ not defined.?')
        with torch.no_grad():
            res = self.vf_theta_(self.params_, *args)
            res = torch.hstack([g.flatten(start_dim=1) for g in res.values()])
        return res
    
    def detach(self) -> None:
        ## Make the parameters not request gradient computation by creating a detached copy.
        ## The copy is shallow, meaning it shares memory, and is tied to parameter updates in the outer loop.
        new_params = {key: tensor.detach() for key, tensor in self.params_.items()}
        new_net = NetWithPartials(self.f_, self.f_x_, self.vf_x_, self.f_xx_, self.vf_xx_, new_params, self.vf_z_)
        return new_net