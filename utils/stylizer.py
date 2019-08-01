import sys
sys.path.append('./utils/')

import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import net

class styleTrans():
    def __init__(self,device,decoder_path=None,transform_path=None,vgg_path=None):
        self.device = device
        self.decoder = net.decoder
        self.vgg = net.vgg
        self.network=net.Net(self.vgg,self.decoder)
        self.sa_module = self.network.sa_module


        self.decoder.eval()
        self.sa_module.eval()
        self.vgg.eval()

        self.decoder.load_state_dict(torch.load(decoder_path))
        self.sa_module.load_state_dict(torch.load(transform_path))
        self.vgg.load_state_dict(torch.load(vgg_path))

        self.norm = nn.Sequential(*list(self.vgg.children())[:1])
        self.enc_1 = nn.Sequential(*list(self.vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(self.vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(self.vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(self.vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(self.vgg.children())[31:44])  # relu4_1 -> relu5_1

        self.norm.to(device)
        self.enc_1.to(device)
        self.enc_2.to(device)
        self.enc_3.to(device)
        self.enc_4.to(device)
        self.enc_5.to(device)
        self.sa_module.to(device)
        self.decoder.to(device)
        print('model init')

    def feat_extractor(self, content, style):
        Content4_1 = self.enc_4(self.enc_3(self.enc_2(self.enc_1(content))))
        Content5_1 = self.enc_5(Content4_1)

        Style4_1 = self.enc_4(self.enc_3(self.enc_2(self.enc_1(style))))
        Style5_1 = self.enc_5(Style4_1)

        content_f=[Content4_1,Content5_1]
        style_f=[Style4_1,Style5_1]
        return content_f, style_f
    
    def stansform(self, content,style,alpha=0.4,interpolation_weights=None):
        assert (0.0 <= alpha <= 1.0)
        content_f, style_f=self.feat_extractor(content, style)
        Fccc = self.sa_module(content_f,content_f)

        if interpolation_weights:
            _, C, H, W = Fccc.size()
            feat = torch.FloatTensor(1, C, H, W).zero_().to(self.device)
            base_feat = self.sa_module(content_f, style_f)
            for i, w in enumerate(interpolation_weights):
                feat = feat + w * base_feat[i:i + 1]
            Fccc=Fccc[0:1]
        else:
            feat = self.sa_module(content_f, style_f)
        feat = feat * alpha + Fccc * (1 - alpha)
        return self.decoder(feat)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform