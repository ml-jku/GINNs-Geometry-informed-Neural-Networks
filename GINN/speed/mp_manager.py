
import atexit
from contextlib import nullcontext
import copy
import multiprocessing as mp
from GINN.speed.timer import Timer
from GINN.speed.dummy_async_res import DummyAsyncResult


def wrapper(func, timer, time_str, arg_tuples, kwargs_dict):
    try:
        with Timer.record(time_str, timer):
            # print(f'Calling {func.__name__} with args {arg_tuples} and kwargs {kwargs_dict}')
            res = func(*arg_tuples, **kwargs_dict)
            return res
    except Exception as e:
        print(f'Exception with traceback: {e}')
        raise e

class MPManager():
    
    def __init__(self, n_workers=0) -> None:
        self.is_mp_on = n_workers > 1
        if self.is_mp_on:
            # use dill for pickling
            # ctx = mp.get_context()
            # ctx.reducer = dill.reducer
            self.manager = mp.Manager()
            print(f'Using {n_workers} workers')
            self.pool = mp.Pool(processes=n_workers)
            atexit.register(self._cleanup_pool)
        self.async_results_dict = {}
        self.epoch = -1
        self.timer = None
                
    def update_epoch(self, epoch):
        self.epoch = epoch

    def set_timer(self, timer):
        self.timer = timer  ## attention: circular reference!

    def get_lock(self):
        if self.is_mp_on:
            return self.manager.Lock()
        return nullcontext()

    def _cleanup_pool(self):
        self.pool.close()
        self.pool.join()
    
    def plot(self, func, fig_key, arg_list=[], kwargs_dict={}):                
        if self.epoch not in self.async_results_dict:
            self.async_results_dict[self.epoch] = []
        
        # Execute the function in parallel if multiprocessing is on
        if self.is_mp_on:
            # deepcopy the args and kwargs to avoid pickling errors
            arg_list = copy.deepcopy(arg_list)
            kwargs_dict = copy.deepcopy(kwargs_dict)
            # ATTENTION: can not pass tensor as argument to multiprocessing
            self.async_results_dict[self.epoch].append(self.pool.apply_async(wrapper, (func, self.timer, fig_key, arg_list, kwargs_dict)))
        else:
            with self.timer.record(f'plot_{func.__name__}'):
                self.async_results_dict[self.epoch].append(DummyAsyncResult(func(*arg_list, **kwargs_dict)))
                
    def metrics(self, func, arg_list=[], kwargs_dict={}):
        
        if self.epoch not in self.async_results_dict:
            self.async_results_dict[self.epoch] = []
            
            
        if self.is_mp_on:
            arg_list = copy.deepcopy(arg_list)
            kwargs_dict = copy.deepcopy(kwargs_dict)
            # ATTENTION: can not pass tensor as argument to multiprocessing
            async_dict_res = self.pool.apply_async(wrapper, (func, self.timer, 'metrics', arg_list, kwargs_dict))
            self.async_results_dict[self.epoch].append(async_dict_res)
        else:
            with self.timer.record(f'metrics_{func.__name__}'):
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
    
    def _pop_iter_results(self, epoch):
        if epoch not in self.async_results_dict:
            raise ValueError(f'Epoch {epoch} not in async_results_dict - did you forget to call exec_plot?')
        
        for result in self.async_results_dict[epoch]:
            res = result.get()
            yield res
        del self.async_results_dict[epoch]
        
    def pop_results_dict(self, epoch):
        '''
        Returns a dict of {fig_label: wandb_img} for all plots in the given epoch.
        After calling this method, the plots for the given epoch are removed from the async_results_dict.  
        '''
        res_dict = {}
        for result in self._pop_iter_results(epoch):
            if result is None:
                continue
            
            if isinstance(result, dict):
                res_dict.update(result)
            elif isinstance(result, tuple):
                fig_label, wandb_img = result
                res_dict[fig_label] = wandb_img
            else:
                raise ValueError(f'Unexpected result type: {type(result)}, result: {result}')
        return res_dict