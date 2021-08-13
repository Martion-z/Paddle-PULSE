#!/usr/bin/env python
# coding: utf-8

# In[4]:


import paddle
from paddle.optimizer import Optimizer

# Spherical Optimizer Class
# Uses the first two dimensions as batch information
# Optimizes over the surface of a sphere using the initial radius throughout
#
# Example Usage:
# opt = SphericalOptimizer(torch.optim.SGD, [x], lr=0.01)

class SphericalOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(parameters=params, **kwargs)
        self.params = params
        self.radii={}
        with paddle.no_grad():
            for param in params:
               self.radii[param]=(param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt()
            #self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}

    @paddle.no_grad()
    def step(self, closure=None):

        loss = self.opt.step()
        for param in self.params:
            param1=paddle.divide(param,(param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param=paddle.multiply(param1,(self.radii[param]))

        return loss

