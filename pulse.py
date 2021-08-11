#!/usr/bin/env python
# coding: utf-8

# In[3]:

import json
from stylegan_paddle import G_synthesis,G_mapping
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import paddle
from loss import LossBuilder
from functools import partial
from drive import open_url
import argparse
import paddle.nn as nn
from paddle.static import InputSpec
from visualdl import LogWriter

class PULSE(paddle.nn.Layer):
    def __init__(self, cache_dir, batch_size, verbose=True):
        super(PULSE, self).__init__()

        self.synthesis = G_synthesis(bt=batch_size)##生成器
        self.verbose = verbose

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok = True)
        if self.verbose: print("Loading Synthesis Network")#输出冗余信息
        #加载模型
        #with open_url("https://drive.google.com/uc?id=1TCViX1YpQyRsklTVYEJwdbmK91vklCo8", cache_dir=cache_dir, verbose=verbose) as f:
        f=paddle.load("styleGan.pdparams")
        self.synthesis.set_dict(f)

        #冻结模型参数
        for param in self.synthesis.parameters():
            param.stop_gradient=True
            
        #定义激活函数
        
        self.lrelu = paddle.nn.LeakyReLU(negative_slope=0.2)

        if Path("gaussian_fit").exists():
            self.gaussian_fit = paddle.load("gaussian_fit")
        else:
            if self.verbose: print("\tLoading Mapping Network")
            mapping = G_mapping()

            with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir=cache_dir, verbose=verbose) as f:
                    mapping.load(f)

            if self.verbose: print("\tRunning Mapping Network")
            with paddle.no_grad():
                paddle.seed(0)
                latent = paddle.randn((1000000,512),dtype=paddle.float32, device="cuda")
                latent_out = paddle.nn.LeakyReLU(5)(mapping(latent))
                self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
                paddle.save(self.gaussian_fit,"gaussian_fit.pt")
                if self.verbose: print("\tSaved \"gaussian_fit.pt\"")

    def forward(self, ref_im,
                seed,
                loss_str,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate0,
                steps,
                lr_schedule,
                save_intermediate,

                **kwargs):

        if seed:
            paddle.seed(seed)
            paddle.cuda.seed(seed)
            paddle.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]

        # Generate latent tensor
        if(tile_latent):
            latentcode = np.random.randn(batch_size, 1, 512)
            latent = paddle.create_parameter(shape=(batch_size, 1, 512),dtype='float32',default_initializer=nn.initializer.Assign(latentcode))      
        else:
            latentcode = np.random.randn(batch_size, 18, 512)
            latent = paddle.create_parameter(shape=(batch_size, 18, 512),dtype='float32',default_initializer=nn.initializer.Assign(latentcode))
            
        


        # Generate list of noise tensors
        noise = [] # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on
        
        for i in range(18):
            # dimension of the ith noise tensor
            res = (batch_size, 1, 2**(i//2+2), 2**(i//2+2))

            if(noise_type == 'zero' or i in [int(layer) for layer in bad_noise_layers.split('.')]):
                new_noise = paddle.zeros(res, dtype=paddle.float32)
                
            elif(noise_type == 'fixed'):
                new_noise = paddle.randn(res, dtype=paddle.float32)
                
            elif (noise_type == 'trainable'):
                new_noise = np.random.randn(batch_size, 1, 2**(i//2+2), 2**(i//2+2))
                
                if (i < num_trainable_noise_layers):
                    new_noise = paddle.create_parameter(shape=res,dtype='float32',default_initializer=nn.initializer.Assign(new_noise))
                    noise_vars.append(new_noise)
                else:
                    new_noise=paddle.to_tensor(new_noise)
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)
        
        var_list = [latent]+noise_vars  ##优化参数

        opt_dict = {
            'sgd': paddle.optimizer.SGD,
            'adam': paddle.optimizer.Adam,
            'sgdm': partial(paddle.optimizer.SGD, momentum=0.9),
            'adamax': paddle.optimizer.Adamax
        }
        opt_func = opt_dict[opt_name]
        
        opt = SphericalOptimizer(opt_func, var_list,learning_rate=learning_rate0)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate0,schedule_func)  ##调整学习率
        
        loss_builder = LossBuilder(ref_im, loss_str, eps)

        min_loss = np.inf
        min_l2 = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None


        if self.verbose: print("Optimizing")
        f = open('log.txt', 'a')
        f.write('------------Optimizing start----------------\n')
        f.close()
        for j in range(steps):
            opt.opt.clear_grad()

              # Duplicate latent in case tile_latent = True
            if (tile_latent):
                latent_in = latent.expand(-1, 18, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            gen_im = (self.synthesis(latent_in, noise)+1)/2
            
            # Calculate Losses
            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss
            f=open('log.txt','a')
            f.write("step")
            f.write(str(j+1))
            f.write('    L2:')
            f.write(str(loss_dict['L2'].numpy()[0]))
            f.write('    GEOCROSS:')
            f.write(str(loss_dict['GEOCROSS'].numpy()[0]))
            f.write('    TOTAL:')
            f.write(str(loss_dict['TOTAL'].numpy()[0]))
            f.write('\n')
            f.close()

            # Save best summary for log
            if(loss < min_loss):
                min_loss = loss  
                best_summary = f'BEST ({j+1}) | '+' | '.join(
                [f'{x}: {y.numpy()[0]}' for x, y in loss_dict.items()])     
                best_im = gen_im.clone()
                
            loss_l2 = loss_dict['L2']

            if(loss_l2 < min_l2):
                min_l2 = loss_l2

            # Save intermediate HR and LR images
            if(save_intermediate):
                yield (best_im.cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))

            
            
            loss.backward()
            opt.step()
            scheduler.step()
                    
        yield (paddle.clip(best_im.clone().cpu().detach(),0, 1),paddle.clip(loss_builder.D(best_im).cpu().detach(),0, 1))
        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose:             
            print(best_summary)
        
        if(min_l2 > eps):
            print("Could not find a face that downscales correctly within epsilon,But save the best image in output directory")





# In[2]:






