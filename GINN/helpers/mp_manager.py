
import atexit
from contextlib import nullcontext
import copy
from math import e
import multiprocessing
import re
from unittest import result

from sympy import N

from utils import do_plot

def wrapper(func, timer_helper, time_str, arg_tuples, kwargs_dict):
    try:
        with timer_helper.record(time_str):
            # print(f'Calling {func.__name__} with args {arg_tuples} and kwargs {kwargs_dict}')
            return func(*arg_tuples, **kwargs_dict)
    except Exception as e:
        # print traceback
        print(f'Exception with traceback: {e}')
        raise e

class DummyAsyncResult():
    def __init__(self, result):
        self.result = result
    def ready(self):
        return True
    def get(self):
        return self.result

class MPManager():
    
    def __init__(self, config) -> None:
        self.config = config      
        n_workers = config.get('num_workers', 0)
        self.is_mp_on = n_workers > 0
        if self.is_mp_on:
            self.manager = multiprocessing.Manager()
            print(f'Using {n_workers} workers')
            self.pool = multiprocessing.Pool(processes=n_workers)
        atexit.register(self._cleanup_pool)
        self.async_results_dict = {}
        self.epoch = -1
        self.timer_helper = None
                
    def update_epoch(self, epoch):
        self.epoch = epoch

    def set_timer_helper(self, timer_helper):
        self.timer_helper = timer_helper  ## attention: circular reference!

    def get_lock(self):
        if self.is_mp_on:
            return self.manager.Lock()
        return nullcontext()

    def _cleanup_pool(self):
        if self.is_mp_on:
            self.pool.close()
            self.pool.join()
    
    def _do_plot_only_every_n_epochs(self):
        if 'plot_every_n_epochs' not in self.config or self.config['plot_every_n_epochs'] is None:
            return True
        return (self.epoch % self.config['plot_every_n_epochs'] == 0) or (self.epoch == self.config['max_epochs'])

    def plot(self, func, fig_key, arg_list=[], kwargs_dict={}, dont_parallelize=False):
        if not do_plot(self.config, self.epoch, key=fig_key):
            return
        
        if not self._do_plot_only_every_n_epochs():
            return
                
        if self.epoch not in self.async_results_dict:
            self.async_results_dict[self.epoch] = []
        
        # print(f'Executing {func.__name__} with args {arg_list} and kwargs {kwargs_dict}')
        # print(f'Parallelization is {"" if self.is_mp_on else "NOT "}on')
        # print('parallelization is', 'on' if self.is_mp_on else 'off')
        
        # Execute the function in parallel if multiprocessing is on
        if not dont_parallelize and self.is_mp_on:
            # deepcopy the args and kwargs to avoid pickling errors
            arg_list = copy.deepcopy(arg_list)
            kwargs_dict = copy.deepcopy(kwargs_dict)
            self.async_results_dict[self.epoch].append(self.pool.apply_async(wrapper, (func, self.timer_helper, fig_key, arg_list, kwargs_dict)))
        else:
            with self.timer_helper.record(f'plot_{func.__name__}'):
                self.async_results_dict[self.epoch].append(DummyAsyncResult(func(*arg_list, **kwargs_dict)))
    
    def add_dummy_async_result(self, result):
        if self.epoch not in self.async_results_dict:
            self.async_results_dict[self.epoch] = []
        self.async_results_dict[self.epoch].append(DummyAsyncResult(result))
    
    def are_plots_available_for_epoch(self, epoch):
        return epoch in self.async_results_dict
        
    def plots_ready_for_epoch(self, epoch):
        if epoch not in self.async_results_dict:
            raise ValueError(f'Epoch {epoch} not in async_results_dict - did you forget to call exec_plot?')
        return all([result.ready() for result in self.async_results_dict[epoch]])
    
    def _iter_plots(self, epoch):
        if epoch not in self.async_results_dict:
            raise ValueError(f'Epoch {epoch} not in async_results_dict - did you forget to call exec_plot?')
        for result in self.async_results_dict[epoch]:
            yield result.get()
        del self.async_results_dict[epoch]
        
    def pop_plots_dict(self, epoch):
        '''
        Returns a dict of {fig_label: wandb_img} for all plots in the given epoch.
        After calling this method, the plots for the given epoch are removed from the async_results_dict.  
        '''
        img_dict = {}
        for result_tuple in self._iter_plots(epoch):
            if result_tuple is None:
                continue
            fig_label, wandb_img = result_tuple
            img_dict[fig_label] = wandb_img
        return img_dict