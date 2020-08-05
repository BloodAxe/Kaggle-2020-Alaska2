import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
import sys
import tensorflow as tf
import jpegio as jio
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer 

### Tensorflow
def get_lr_schedule(id_num):
    if id_num == 0:
    # From scratch
        return {'boundaries': [20000, 200000],
                'values': [1e-4, 1e-3, 1e-4],
                'max_iter': 300000}
    elif id_num == 1:
    # From QF75
        return {'boundaries': [10000, 120000, 140000],
                'values': [1e-4, (1e-4)/2, (1e-4)/4, (1e-4)/8],
                'max_iter': 160000}
    elif id_num == 2:
    # noPC 
        return {'boundaries': [20000, 80000, 120000, 160000],
                'values': [1e-4, 1e-3, (1e-3)/10, (1e-3)/10/5, (1e-3)/10/10],
                'max_iter': 200000}

class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
        
### Pytorch
        
def get_optimizer(optimizer_name):
    import torch
    if optimizer_name.lower() == 'sgd':
        return torch.optim.SGD
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW    
    

    