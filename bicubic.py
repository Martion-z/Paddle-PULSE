#!/usr/bin/env python
# coding: utf-8

# In[38]:


import paddle
from paddle import nn
from paddle.nn import functional as F
import math
import paddle.fluid as fluid

class BicubicDownSample(nn.Layer):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = paddle.abs(x)

        if abs_x <= 1.:
            return (a + 2.) * paddle.pow(abs_x, 3.) - (a + 3.) * paddle.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * paddle.pow(abs_x, 3) - 5. * a * paddle.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return paddle.to_tensor(0.0, dtype=paddle.float32)

    def __init__(self, factor=4, cuda=True, padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        tk=[]
        for i in range(size):
            
            a=self.bicubic_kernel((i-paddle.floor(paddle.to_tensor(size / 2,dtype=paddle.float32))+0.5)/factor)

            tk.append(a)
              
        k=paddle.to_tensor(tk,dtype=paddle.float32)

        '''
        k = paddle.to_tensor([self.bicubic_kernel((i - paddle.floor(paddle.to_tensor(size / 2,dtype=paddle.float32)) + 0.5) / factor)
                          for i in range(size)], dtype=paddle.float32)
        '''
        k = k / paddle.sum(k)

        k1 = paddle.reshape(k, shape=(1, 1, size, 1))
        self.k1 = paddle.concat([k1, k1, k1], 0)
        k2 = paddle.reshape(k, shape=(1, 1, 1, size))
        self.k2 = paddle.concat([k2, k2, k2], 0)
        self.cuda = '.cuda' if cuda else ''
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):

        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)

        filters1 = self.k1
        filters2 = self.k2
        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        
        # apply mirror padding
        if nhwc:
            x = paddle.transpose(paddle.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW

        # downscaling performed by 1-d convolution
        x = F.pad(x, [0, 0, pad_top, pad_bottom], self.padding)
        
        x = F.conv2d(x, weight=filters1, stride=(stride, 1), groups=3)

        if clip_round:
            x = paddle.clip(paddle.round(x), 0.0, 255.)

        x = F.pad(x, [pad_left, pad_right, 0, 0], self.padding)
        x = F.conv2d(x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = paddle.clip(paddle.round(x), 0.0, 255.)

        if nhwc:
            x = paddle.transpose(paddle.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('paddle.ByteTensor'.format(self.cuda))
        else:
            return x


