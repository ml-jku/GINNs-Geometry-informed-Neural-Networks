import torch
import torch.nn as nn
import inspect
from typing import Any, Dict


class ModelFactory:
    @staticmethod
    def save(model: nn.Module, path: str):
        # Get the constructor arguments
        init_params = ModelFactory._get_init_params(model)
        state = {
            'state_dict': model.state_dict(),
            'init_params': init_params
        }
        torch.save(state, path)

    @staticmethod
    def load(model_class: nn.Module, path: str, map_location=None, **kwargs) -> nn.Module:
        if 'map_location' is not None:
            state = torch.load(path, map_location=map_location)
        else:
            state = torch.load(path)
        init_params = state['init_params']

        # Merge init_params with kwargs, with kwargs taking precedence
        init_params.update(kwargs)

        model = model_class(**init_params)
        model.load_state_dict(state['state_dict'])
        return model

    @staticmethod
    def _get_init_params(model: nn.Module) -> Dict[str, Any]:
        # Get the constructor signature
        sig = inspect.signature(model.__init__)
        init_params = {}

        # Iterate over the parameters of the constructor
        for param in sig.parameters.values():
            if param.name != 'self' and param.name in model.__dict__:
                init_params[param.name] = model.__dict__[param.name]

        return init_params