"""
record time cost, epoch, step, exp_name, ckpt_path and best value.
"""
import time

class Dispatcher(object):
    def __init__(self, step, epoch, exp_name, ckpt_path):
        self.time_start()
        self.step = step
        self.epoch = epoch
        self.exp_name = exp_name
        self.ckpt_path = ckpt_path
        self.best_dict = {'CC':100.0, 'NUS':100.0, 'Cube':100.0}

    def time_cost(self):
        end = time.time()
        cost = end - self.start
        self.start = end
        return cost 
    
    def time_start(self):
        self.start = time.time()