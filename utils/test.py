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
sys.path.append('../')
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
from stylizer import styleTrans,test_transform, Reinhard_color_transfer
from matting import mat

torch.set_grad_enabled(False)

# const for style trans
content="./imgs/girl.jpg"
style = "./imgs/3.jpg,./imgs/3_patch.jpg"
vgg_path='./pretrained/style_models/vgg_normalised.pth'
decoder_path='./pretrained/style_models/decoder_iter_92000.pth'
transform_path='./pretrained/style_models/sa_module_iter_92000.pth'

# Additional options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path='./output'

# Advanced options
preserve_color='store_true'
alpha=0.8
style_interpolation_weights=''
preserve_color=False
do_interpolation = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert content
assert style

# Path preprocessing
if os.path.isdir(style):
    style_paths = [os.path.join(style, f) for f in
                   os.listdir(style)]
else:
    style_paths = style.split(',')
    if len(style_paths) == 1:
        style_paths = [style]
    else:
        do_interpolation = True
if not os.path.exists(output_path):
    os.mkdir(output_path)

content_tf = test_transform(content_size,crop)
style_tf = test_transform(style_size,crop)
# preprocessing the image
# style = Image.open(style)
# csize = content.size


# _content = content_tf(content)
# _style = style_tf(style)

# _style = _style.to(device).unsqueeze(0)
# _content = _content.to(device).unsqueeze(0)
transformer = styleTrans(device=device,vgg_path=vgg_path,
                        transform_path=transform_path,
                        decoder_path=decoder_path)

# with torch.no_grad():
#     content_trans = transformer.stansform(content=_content,style=_style,alpha=alpha)
# content_trans = content_trans.cpu()

# cal mask
cm = mat(use_gpu=True)
img = cv2.cvtColor(np.asarray(Image.open(content)),cv2.COLOR_RGB2BGR)  

mask = cm.mat_processing(img,512,0.9)
mask = Image.fromarray(mask).convert('L')
im = Image.new("RGB", mask.size)
im.paste(Image.open(content),mask=mask)
im.save('./result/mask.png')

patch_percent = 0.8
content = Image.fromarray(cv2.cvtColor(Reinhard_color_transfer(content, style.split(',')[0]), cv2.COLOR_BGR2RGB))

_style = torch.stack([style_tf(Image.open(p)) for p in style_paths])
_content = content_tf(content).unsqueeze(0).expand_as(_style)
_style = _style.to(device)
_content = _content.to(device)
with torch.no_grad():
    content_trans = transformer.stansform(content=_content,style=_style,interpolation_weights=[patch_percent,1-patch_percent],alpha=alpha)
    content_trans = content_trans.cpu()

grid = make_grid(content_trans, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
output_ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
content_s = Image.fromarray(output_ndarr)
content_s.save('./result/content_transfered.png')

bg = Image.open(style.split(',')[0])
content_s = content_s.resize((256, 256))
# locx = bg.size[0]-content_s.size[0]
# locy = bg.size[1]-content_s.size[1]
locx = int(bg.size[0] / 2) - 200
locy = int(bg.size[1] / 2) - 400

bg.paste(content_s,box=(locx,locy),mask=mask.resize((256, 256)))

bg.save('./result/mask_content_style.png')
print('Done')





