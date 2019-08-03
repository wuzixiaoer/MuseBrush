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

class style_transfer():
    def __init__(self):
        # const for style trans
        self.content = ''
        self.style = ''
        self.patch = ''
        self.vgg_path='./pretrained/style_models/vgg_normalised.pth'
        self.decoder_path='./pretrained/style_models/decoder_iter_92000.pth'
        self.transform_path='./pretrained/style_models/sa_module_iter_92000.pth'
        # Additional options
        self.crop = None
        self.save_ext = '.jpg'
        self.output_path = './output'
        # Advanced options
        self.preserve_color = 'store_true'
        self.alpha = 0.8
        self.gl_ratio = 0.8
        self.style_interpolation_weights = ''
        self.preserve_color = False
        self.do_interpolation = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # transform
        self.trans = None
        # load model
        self.transformer = styleTrans(device=self.device, vgg_path=self.vgg_path,
                                      transform_path=self.transform_path,
                                      decoder_path=self.decoder_path)
        self.matting = mat(use_gpu=True)
    
    def img_matting(self, loc):  # 分割
        content = Image.open(self.content)
        img = cv2.cvtColor(np.asarray(content), cv2.COLOR_RGB2BGR)
        mask = self.matting.mat_processing(img, 512, 0.9)
        mask = Image.fromarray(mask).convert('L')
        im = Image.new('RGB', mask.size)
        im.paste(content, box=(loc[0], loc[1]), mask=mask)
        # paste to style image
        style_w, style_h = self.style.size
        if im.size[0] > style_w:
            scale = style_w / im_w
            im = im.resize((int(im_w * scale), int(im_h * scale)), Image.ANTIALIAS)
            mask = mask.resize((int(im_w * scale), int(im_h * scale)), Image.ANTIALIAS)
        if im.size[1] > style_h:
            scale = style_h / im_h
            im = im.resize((int(im_w * scale), int(im_h * scale)), Image.ANTIALIAS)
            mask = mask.resize((int(im_w * scale), int(im_h * scale)), Image.ANTIALIAS)
        res = self.style
        res.paste(im, mask=mask)
        return res, mask

    def img_transfer(self, content):  # 风格迁移
        patch_trans = test_transform(self.style.size, None)
        _style = torch.stack([self.trans(self.style), patch_trans(self.patch)])
        _content = self.trans(content).unsqueeze(0).expand_as(_style)
        _style = _style.to(self.device)
        _content = _content.to(self.device)
        with torch.no_grad():
            content_trans = self.transformer.stansform(content=_content, style=_style,
                                                  interpolation_weights=[self.gl_ratio, 1-self.gl_ratio],
                                                  alpha=self.alpha)
            content_trans = content_trans.cpu()
        return content_trans

    def transfer(self, content, style_dict): # 风格迁移pipeline, content=path, style=path
        self.content = content
        self.style = Image.open(style_dict['style_src'])
        self.trans = test_transform(self.style.size, None)
        self.patch = Image.open(style_dict['patch_src'])
        self.alpha = style_dict['alpha']
        self.gl_ratio = style_dict['gl_ratio']
        loc = style_dict['loc']

        im, mask = self.img_matting(loc)
        print(type(im))
        # self.content = Image.fromarray(cv2.cvtColor(Reinhard_color_transfer(self.content, self.style), cv2.COLOR_BGR2RGB))
        content_trans = self.img_transfer(im)
        final_result = self.style.paste(content_trans, mask=mask)
        return final_result
