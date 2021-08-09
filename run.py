#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pulse import PULSE
from paddle.io import Dataset, DataLoader
import paddle
from pathlib import Path
from PIL import Image
from math import log10, ceil
import argparse
import numpy as np
from paddle.static import InputSpec

# In[8]:


class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        image=Image.open(img_path)
        image = paddle.vision.transforms.to_tensor(image)
        if(self.duplicates == 1):
            return image,img_path.stem
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}"


# In[9]:



parser = argparse.ArgumentParser(description='pulse')

#I/O arguments
parser.add_argument('-input_dir', type=str, default='input', help='input data directory')
parser.add_argument('-output_dir', type=str, default='output2', help='output data directory')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-duplicates', type=int, default=1, help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')

#PULSE arguments
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS", help='Loss function to use')
parser.add_argument('-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate0', type=float, default=0.04, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')

kwargs = vars(parser.parse_args())

dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])

model = PULSE(cache_dir=kwargs["cache_dir"],batch_size=kwargs["batch_size"])
model = paddle.DataParallel(model)

#toPIL = torchvision.transforms.ToPILImage()

for ref_im, ref_im_name in dataloader:
    if(kwargs["save_intermediate"]):
        padding = ceil(log10(100))
        for i in range(kwargs["batch_size"]):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j,(HR,LR) in enumerate(model(ref_im,**kwargs)):
            for i in range(kwargs["batch_size"]):
                out_hr = HR[i].cpu().detach().clamp(0, 1).numpy().transpose((1, 2, 0))*255
                out_hr = Image.fromarray(out_hr.astype(np.uint8))
                out_hr.save(int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                in_lr = LR[i].cpu().detach().clamp(0, 1).numpy().transpose((1, 2, 0))*255
                in_lr = Image.fromarray(in_lr.astype(np.uint8))
                in_lr.save(int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                '''
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")

                output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
                output = Image.fromarray(output.astype(np.uint8))
                '''
    else:
        #out_im = model(ref_im,**kwargs)
        for j,(HR,LR) in enumerate(model(ref_im,**kwargs)):
            for i in range(kwargs["batch_size"]):
                out_hr = paddle.clip(HR[i].cpu().detach(), 0, 1).numpy().transpose((1, 2, 0))*255
                #out_hr = HR[i].cpu().detach().numpy().transpose((1, 2, 0))
                out_hr = Image.fromarray(out_hr.astype(np.uint8))
                out_hr.save(out_path / f"{ref_im_name[i]}.png")
                #toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                #    out_path / f"{ref_im_name[i]}.png")

