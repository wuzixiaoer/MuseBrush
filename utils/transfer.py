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
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import cv2
from tqdm import tqdm
from config import cfg
import net as net
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
import copy

module_path = os.path.dirname(__file__) 
class style_transfer():
    def __init__(self):
        # const for style trans
        self.content = ''
        self.style = ''
        self.patch = ''
        
        self.vgg_path = module_path + '/pretrained/style_models/vgg_normalised.pth'
        self.decoder_path =  module_path + '/pretrained/style_models/decoder_iter_92000.pth'
        self.transform_path = module_path + '/pretrained/style_models/sa_module_iter_92000.pth'
        # Additional options
        self.crop = None
        self.save_ext = '.jpg'
        self.output_path = './output'
        # Advanced options
        self.preserve_color = 'store_true'
        self.alpha = 1
        self.gl_ratio = 0.9
        self.style_interpolation_weights = ''
        self.preserve_color = False
        self.do_interpolation = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # transform
        self.trans = test_transform((512,512), None)
        # load model
        self.transformer = styleTrans(device=self.device, vgg_path=self.vgg_path,
                                      transform_path=self.transform_path,
                                      decoder_path=self.decoder_path)
        self.matting = mat(use_gpu=True)
    
    def img_matting(self, loc, hsize):  # 分割
        img = cv2.cvtColor(np.asarray(self.content), cv2.COLOR_RGB2BGR)
        mask = self.matting.mat_processing(img, 512, 0.9)
        mask = Image.fromarray(mask).convert('L')
        im = Image.new('RGB', mask.size)
        im.paste(self.content, mask=mask)
        # paste to style image
        # style_w, style_h = self.style.size
        # if im.size[0] > style_w:
        #     ratio = style_w / im.size[0]
        #     im = ratio_resize(im, ratio)
        #     mask = ratio_resize(mask, ratio)
        # if im.size[1] > style_h:
        #     ratio = style_h / im.size[1]
        #     im = ratio_resize(im, ratio)
        #     mask = ratio_resize(mask, ratio)
        # res = copy.deepcopy(self.style)
        # res.paste(im, mask=mask)
        return im, mask

    def img_transfer(self, content):  # 风格迁移
        content = content.resize(self.style.size, Image.ANTIALIAS)
        _style = torch.stack([self.trans(self.style), self.trans(self.patch)])
        _content = self.trans(content).unsqueeze(0).expand_as(_style)
        
        _style = _style.to(self.device)
        _content = _content.to(self.device)
        with torch.no_grad():
            content_trans = self.transformer.stansform(content=_content, style=_style,
                                                  interpolation_weights=[self.gl_ratio, 1-self.gl_ratio],
                                                  alpha=self.alpha)
            content_trans = content_trans.cpu()
        # grid = make_grid(content_trans, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
        # output_ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # print(np.shape(output_ndarr))
        content_s = transforms.ToPILImage()(content_trans[0]).convert('RGB')
        content_s.save(module_path + '/result/conent_s.png')
        return content_s

    def transfer(self, content, style_dict): # 风格迁移pipeline, content=path, style=path
        self.content = Image.open(content)
        self.style = Image.open(style_dict['style_src'])
        
        self.patch = Image.open(style_dict['patch_src'])
        # self.patch = self.patch.resize(self.style.size, Image.ANTIALIAS)
        self.alpha = style_dict['alpha']
        self.gl_ratio = style_dict['gl_ratio']
        hsize = style_dict['hsize']
        loc = style_dict['loc']

        im, mask = self.img_matting(loc, hsize)
        mask.save(module_path + '/result/mask.png')
        im.save(module_path + '/result/im.png')
        print(mask.size)
        self.content = Image.fromarray(cv2.cvtColor(Reinhard_color_transfer(self.content, self.style), cv2.COLOR_BGR2RGB))
        content_s = self.img_transfer(self.content)
        content_s = content_s.resize(mask.size, Image.ANTIALIAS)
        # resize
        bbox = mask.getbbox()
        mask = mask.crop(bbox)
        content_s = content_s.crop(bbox)
        ratio = hsize / mask.size[1]
        mask = ratio_resize(mask, ratio)
        content_s = ratio_resize(content_s, ratio)
        
        final_result = copy.deepcopy(self.style)
        final_result.paste(content_s, loc, mask=mask)
        final_result.save(module_path + '/result/result.png')
        return final_result


def ratio_resize(img, scale):
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)