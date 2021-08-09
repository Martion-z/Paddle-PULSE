#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[343]:


import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from collections import OrderedDict
import pickle
import numpy as np


# In[344]:


class MyLinear(nn.Layer):
    def __init__(self, input_size, output_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size**(-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        w=np.random.randn(output_size,input_size)*init_std
        self.weight = paddle.create_parameter(shape=[output_size,input_size ],dtype='float32',default_initializer=nn.initializer.Assign(w))
        #print(self.weight)
        if bias:    
            self.bias=paddle.create_parameter(shape=[output_size],dtype='float32',default_initializer=nn.initializer.Constant(0))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        #print("GMYline", F.linear(x, (self.weight * self.w_mul).t(), bias))
        return F.linear(x, (self.weight * self.w_mul).t(), bias)



# In[345]:


class MyConv2d(nn.Layer):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True,
                 intermediate=None, upscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        he_std = gain * (input_channels * kernel_size **
                         2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        w=np.random.randn(output_channels, input_channels, kernel_size, kernel_size)*init_std
        self.weight = paddle.create_parameter(shape=[output_channels, input_channels, kernel_size, kernel_size],dtype='float32',default_initializer=nn.initializer.Assign(w))

        if bias:
            self.bias=paddle.create_parameter(shape=[output_channels],dtype='float32',default_initializer=nn.initializer.Constant(0))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = paddle.transpose(w,perm=[1,0,2,3])
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w,[1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv2d_transpose(
                x, w, stride=2, padding=int((w.shape[-1]-1)//2))
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=int(self.kernel_size//2))
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None,
                         padding=int(self.kernel_size//2))

        if self.intermediate is not None:
            x = self.intermediate(x)
        if bias is not None:
            x = x + paddle.reshape(bias,shape=(1, -1, 1, 1))
        return x
    


# In[346]:


class NoiseLayer(nn.Layer):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = paddle.create_parameter(shape=[channels],dtype='float32',default_initializer=nn.initializer.Constant(0))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = paddle.randn(shape=[x.shape[0], 1, x.shape[2], x.shape[3]]) 
            #print(noise)    
        elif noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
            #print("2",noise)
        x = x + paddle.reshape(self.weight,shape=(1, -1, 1, 1))* noise
        return x


# In[347]:


class StyleMod(nn.Layer):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        #print(x.size(1))
        shape = [-1, 2, x.shape[1], 1, 1]
        style = paddle.reshape(style,shape=[-1, 2, x.shape[1], 1, 1])  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


# In[348]:


class PixelNormLayer(nn.Layer):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * paddle.rsqrt(paddle.mean(x**2, dim=1, keepdim=True) + self.epsilon)


# In[349]:


class BlurLayer(nn.Layer):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = [1, 2, 1]
        #kernel = paddle.to_tensor(kernel, dtype='float32')
        #kernel = kernel[:, None] * kernel[None, :]
        #kernel = kernel[None, None]
        kernel = [[[[1., 2., 1.],
          [2., 4., 2.],
          [1., 2., 1.]]]]
        kernel = paddle.to_tensor(kernel, dtype='float32')
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        #print(x.size(1))
        kernel = paddle.expand(self.kernel,shape=(x.shape[1], -1, -1, -1))
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.shape[2]-1)/2),
            groups=x.shape[1]
        )
        return x


# In[350]:


'''
kernel = [1, 2, 1]
kernel = kernel[None, None]
kernel = paddle.to_tensor(kernel, dtype='float32')
print(kernel)
print(kernel[:, None])
kernel = kernel[:, None] * kernel[None, :]
'''


# In[351]:

'''
def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3],
                   1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(
            shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x
'''

# In[352]:


class Upscale2d(nn.Layer):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = paddle.expand(paddle.reshape(x,(shape[0], shape[1], shape[2], 1, shape[3], 1)),(-1, -1, -1, self.factor, -1, self.factor))
            x = paddle.reshape(x,(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3]))
        return x
        #return upscale2d(x, factor=self.factor, gain=self.gain)


# In[353]:


class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (paddle.Relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [
            ('pixel_norm', PixelNormLayer()),
            ('dense0', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense0_act', act),
            ('dense1', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', act),
            ('dense2', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', act),
            ('dense3', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', act),
            ('dense4', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', act),
            ('dense5', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', act),
            ('dense6', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', act)
        ]
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        x = super().forward(x)
        return x


# In[354]:


class LayerEpilogue(nn.Layer):
    """Things to do at the end of each layer."""

    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        layers = []
        if use_noise:
            self.noise = NoiseLayer(channels)
        else:
            self.noise = None
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNormLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2D(channels,weight_attr=False,bias_attr=False)))
        
        
        self.top_epi = nn.Sequential(*layers)
        if use_styles:
            self.style_mod = StyleMod(
                dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None, noise_in_slice=None):
        if(self.noise is not None):
            x = self.noise(x, noise=noise_in_slice)
        x = self.top_epi(x)

        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


# In[355]:


class InputBlock(nn.Layer):
    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer,bt):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = paddle.create_parameter(shape=[1, nf, 4, 4],dtype='float32',default_initializer=nn.initializer.Constant(1))
            self.bias = paddle.create_parameter(shape=[nf],dtype='float32',default_initializer=nn.initializer.Constant(1))
        else:
            # tweak gain to match the official implementation of Progressing GAN
            self.dense = MyLinear(dlatent_size, nf*16,
                                  gain=gain/4, use_wscale=use_wscale)
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.batch_size=bt

    def forward(self, dlatents_in_range, noise_in_range):
        batch_size =self.batch_size
        #
        if self.const_input_layer:
            #print("Pself.const", self.const)
            x = paddle.expand(self.const,shape=(batch_size, -1, -1, -1))
            x = x + paddle.reshape(self.bias,shape=(1, -1, 1, 1))
        else:
            #x = self.dense(dlatents_in_range[:, 0]).view(
            #    batch_size, self.nf, 4, 4)
            x=paddle.reshape(self.dense(dlatents_in_range[:, 0]),shape=(batch_size, self.nf, 4, 4))
        #print("Ginout", x)
        x = self.epi1(x, dlatents_in_range[:, 0], noise_in_range[0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1], noise_in_range[1])

        return x


# In[356]:


class GSynthesisBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                 intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels,
                              kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)

    def forward(self, x, dlatents_in_range, noise_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0], noise_in_range[0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1], noise_in_range[1])
        return x


# In[357]:


class G_synthesis(nn.Layer):
    def __init__(self,
                 # Disentangled latent (W) dimensionality.
                 dlatent_size=512,
                 num_channels=3,            # Number of output color channels.
                 resolution=1024,         # Output resolution.
                 # Overall multiplier for the number of feature maps.
                 fmap_base=8192,
                 # log2 feature map reduction when doubling the resolution.
                 fmap_decay=1.0,
                 # Maximum number of feature maps in any layer.
                 fmap_max=512,
                 use_styles=True,         # Enable style inputs?
                 const_input_layer=True,         # First layer is a learned constant?
                 use_noise=True,         # Enable noise inputs?
                 # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
                 randomize_noise=True,
                 nonlinearity='lrelu',      # Activation function: 'relu', 'lrelu'
                 use_wscale=True,         # Enable equalized learning rate?
                 use_pixel_norm=False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm=True,         # Enable instance normalization?
                 # Data type to use for activations and outputs.
                 dtype=paddle.float32,
                 # Low-pass filter to apply when resampling activations. None = no filtering.
                 blur_filter=[1, 2, 1],
                 bt=1
                 ):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        act, gain = {'relu': (nn.ReLU, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1
        torgbs = []
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res-1)
            name = '{s}x{s}'.format(s=2**res)
            if res == 2:
                blocks.append((name,
                               InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act,bt)))

            else:
                blocks.append((name,
                               GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1,
                              gain=1, use_wscale=use_wscale)
        self.blocks = nn.LayerDict(OrderedDict(blocks))
        self.batch_size=bt

    def forward(self, dlatents_in, noise_in):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        batch_size = self.batch_size
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2*i:2*i+2], noise_in[2*i:2*i+2])
            else:
                x = m(x, dlatents_in[:, 2*i:2*i+2], noise_in[2*i:2*i+2])
        rgb = self.torgb(x)
        return rgb


