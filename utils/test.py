# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
import sys
#sys.path.append('../')
from models import ModelBuilder, SegmentationModule
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import cv2
from tqdm import tqdm
from config import cfg
import net
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import pandas as pd
from PIL import Image,ImageFilter
from stylizer import styleTrans,test_transform
from genMask import calmask

cfg.merge_from_file("./config/ade20k-resnet50dilated-ppm_deepsup.yaml")

# const for style trans
content="./imgs/li.jpg"
style = "./imgs/brushstrokes.jpg"
vgg_path='./pretrained/style_models/vgg_normalised.pth'
decoder_path='./pretrained/style_models/decoder_iter_100000.pth'
transform_path='./pretrained/style_models/sa_module_iter_100000.pth'

# Additional options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path='./output'

# Advanced options
preserve_color='store_true'
alpha=0.6
style_interpolation_weights=''
preserve_color=False
do_interpolation = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert content
assert style

content_tf = test_transform(content_size,crop)
style_tf = test_transform(style_size,crop)
# preprocessing the image
content = Image.open(content)
style = Image.open(style)
csize = content.size

"""
ssize = style.size
box = (ssize[0]-csize[0],ssize[1]-csize[1],ssize[0],ssize[1])
style = style.crop(box)
"""

_content = content_tf(content)
_style = style_tf(style)

_style = _style.to(device).unsqueeze(0)
_content = _content.to(device).unsqueeze(0)
transformer = styleTrans(device=device,vgg_path=vgg_path,
                        transform_path=transform_path,
                        decoder_path=decoder_path)

with torch.no_grad():
    content_trans = transformer.stansform(content=_content,style=_style,alpha=alpha)
content_trans = content_trans.cpu()


# cal mask
cm = calmask(cfg,gpu=0)
img = cv2.cvtColor(np.asarray(content),cv2.COLOR_RGB2BGR)  

mask = cm.inference(img=img)
_max = pd.value_counts(mask.flatten()).keys()[0]
mask = np.where(mask == _max, 255, 0)
mask = Image.fromarray(mask.astype(np.uint8)).convert('L')
mask.save('./mask.png')



grid = make_grid(content_trans, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
output_ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
content_s = Image.fromarray(output_ndarr)
content_s.save('./ss.png')

bg = style
locx = bg.size[0]-content_s.size[0]
locy = bg.size[1]-content_s.size[1]

bg.paste(content_s,box=(locx,locy),mask=mask)

bg.save('./mask.png')







