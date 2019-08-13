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
        self.content = None
        self.style = None
        self.patch = None
        
        self.vgg_path = module_path + '/pretrained/style_models/vgg_normalised.pth'
        self.decoder_path =  module_path + '/pretrained/style_models/decoder_iter_180000.pth'
        self.transform_path = module_path + '/pretrained/style_models/sa_module_iter_180000.pth'
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
        self.trans = test_transform((512, 512), None)
        # load model
        self.transformer = styleTrans(device=self.device, vgg_path=self.vgg_path,
                                      transform_path=self.transform_path,
                                      decoder_path=self.decoder_path)
        self.matting = mat(use_gpu=True)
    
    def img_matting(self):  # 分割
        img = cv2.cvtColor(np.asarray(self.content), cv2.COLOR_RGB2BGR)
        mask = self.matting.mat_processing(img, 512, 0.9)
        mask = Image.fromarray(mask).convert('L')
        im = Image.new('RGB', mask.size)
        im.paste(self.content, mask=mask)
        # im.save(module_path + '/result/im.png')
        return im, mask

    def img_transfer(self, content, mask):  # 风格迁移
        # content = content.resize(self.style.size, Image.ANTIALIAS)
        style = copy.deepcopy(self.style)
        patch = copy.deepcopy(self.patch)
        style = style.resize(content.size)
        patch = patch.resize(content.size)
        # print(content.size)
        _style = torch.stack([self.trans(style), self.trans(patch)])
        _content = self.trans(content).unsqueeze(0).expand_as(_style)
        _style = _style.to(self.device)
        _content = _content.to(self.device)
        with torch.no_grad():
            content_trans = self.transformer.stansform(content=_content, style=_style,
                                                  interpolation_weights=[self.gl_ratio, 1-self.gl_ratio],
                                                  alpha=self.alpha, mask=mask)
            content_trans = content_trans.cpu()
        grid = make_grid(content_trans, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
        output_ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # print(np.shape(output_ndarr))
        # content_s = transforms.ToPILImage()(content_trans[0]).convert('RGB')
        content_s = Image.fromarray(output_ndarr)
        return content_s

    def transfer(self, content, style_dict): # 风格迁移pipeline, content=path, style=path
        self.content = content
        self.style = Image.open(style_dict['style_src'])
        
        self.patch = Image.open(style_dict['patch_src'])
        # self.patch = self.patch.resize(self.style.size, Image.ANTIALIAS)
        self.alpha = style_dict['alpha']
        self.gl_ratio = style_dict['gl_ratio']
        hsize = style_dict['hsize']
        loc = style_dict['loc']

        im, mask = self.img_matting()
        print('matting done')
        mask.save(module_path + '/result/mask.png')
        # self.content = Image.fromarray(cv2.cvtColor(Reinhard_color_transfer(self.content, self.style), cv2.COLOR_BGR2RGB))
        # print('color transfer done')
        content_s = self.img_transfer(self.content, mask)
        torch.cuda.empty_cache()
        print('style transfer done')
        content_s.save(module_path + '/result/cs.png')
        content_s = content_s.resize(mask.size)
        # content_s.save(module_path + '/result/cs.png')
        # resize
        bbox = mask.getbbox()
        mask = mask.crop(bbox)
        content_s = content_s.crop(bbox)
        # content_s.save(module_path + '/result/cs.jpg')
        ratio = hsize / mask.size[1]
        mask = ratio_resize(mask, ratio)
        mask = mask.convert('L')
        content_s = ratio_resize(content_s, ratio)
        locx = loc[0] - int(content_s.size[0] / 2)
        locy = loc[1] - int(content_s.size[1] / 2)
        # print(content_s.size, mask.size)
        bg = style_dict['bg']
        if bg is not None:
            final_result = Image.open(bg)
        else:
            final_result = self.style
        final_result.paste(content_s, (locx, locy), mask=mask)
        # final_result.save(module_path + '/result/result.jpg')
        return final_result


def ratio_resize(img, scale):
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img
