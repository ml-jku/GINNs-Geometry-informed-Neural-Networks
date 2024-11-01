
import atexit
from contextlib import nullcontext
import copy
import multiprocessing as mp
import torch
from util.misc import do_plot
from GINN.speed.dummy_async_res import DummyAsyncResult

def contains_tensor(obj):
    if isinstance(obj, list):
        return any(contains_tensor(item) for item in obj)
    elif isinstance(obj, dict):
        return any(contains_tensor(value) for value in obj.values())
    else:
        return torch.is_tensor(obj)


def wrapper(func, timer_helper, time_str, arg_tuples, kwargs_dict):
    try:
        with timer_helper.record(time_str):
            # print(f'Calling {func.__name__} with args {arg_tuples} and kwargs {kwargs_dict}')
            res = func(*arg_tuples, **kwargs_dict)
            return res
    except Exception as e:
        print(f'Exception with traceback: {e}')
        raise e

class DummyMPManager():
    
    def __init__(self, **kwargs) -> None:
        pass   
                
    def update_epoch(self, epoch):
        pass

    def set_timer_helper(self, timer_helper):
        pass

    def get_lock(self):
        pass

    def _cleanup_pool(self):
        pass
    
    def _do_plot_only_every_n_epochs(self):
        pass

    def plot(self, func, fig_key, arg_list=[], kwargs_dict={}):
        pass
                
    def metrics(self, func, arg_list=[], kwargs_dict={}):
        pass
    
    def add_dummy_async_result(self, result):
        pass
    
    def are_plots_available_for_epoch(self, epoch):
        pass
        
    def plots_ready_for_epoch(self, epoch):
        pass
        
    def _pop_iter_results(self, epoch):
        pass
        
    def pop_results_dict(self, epoch):
        pass