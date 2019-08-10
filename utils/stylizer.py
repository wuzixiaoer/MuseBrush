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
import cv2
import numpy as np

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
    
    def stansform(self, content, style, alpha=0.4, interpolation_weights=None, mask=None):
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
        mask = np.asarray(mask)
        mask = cv2.resize(mask,(feat.shape[3],feat.shape[2]))
        mask = mask.reshape(1,1,mask.shape[0],mask.shape[1])
        mask = torch.from_numpy(mask).to(self.device,dtype=torch.float32)
        mask = mask / 255
        feat = feat * alpha + Fccc * (1 - alpha)
        feat = feat * mask + Fccc * (1-mask)
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

def get_avg_std(image): # 得到均值和标准差
    avg = []
    std = []
    image_avg_l = np.mean(image[:,:,0])
    image_std_l = np.std(image[:,:,0])
    image_avg_a = np.mean(image[:,:,1])
    image_std_a = np.std(image[:,:,1])
    image_avg_b = np.mean(image[:,:,2])
    image_std_b = np.std(image[:,:,2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg,std)

def Reinhard_color_transfer(content, style):
    #src = cv2.imread(content)
    src = cv2.cvtColor(np.asarray(content),cv2.COLOR_RGB2BGR)  
    src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    des = cv2.cvtColor(np.asarray(style),cv2.COLOR_RGB2BGR)  
    #des = cv2.imread(style)
    des = cv2.cvtColor(des, cv2.COLOR_BGR2LAB)
    src_avg, src_std = get_avg_std(src)
    des_avg, des_std = get_avg_std(des)
    height,width,channel = src.shape
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                t = src[i,j,k]
                t = (t - src_avg[k])*(des_std[k] / src_std[k]) + des_avg[k]
                t = 0 if t < 0 else t
                t = 255 if t > 255 else t
                src[i,j,k] = t
    src = cv2.cvtColor(src, cv2.COLOR_LAB2BGR)
    return src
