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
# Our libs
from utils.dataset import TestDataset
from utils.models import ModelBuilder, SegmentationModule
#from utils import colorEncode, find_recursive, setup_logger
from utils.lib.nn import user_scattered_collate, async_copy_to
from utils.lib.utils import as_numpy
import cv2
from tqdm import tqdm
from utils.config import cfg
from PIL import Image,ImageFilter
from torchvision import transforms
import pandas as pd

class calmask():
    def __init__(self,cfg,gpu=0):
        print(torch.cuda.get_device_name(0))
        torch.cuda.set_device(gpu)
        builder = ModelBuilder()
        net_encoder = builder.build_encoder(
            arch="resnet50dilated",
            fc_dim=2048,
            weights="./utils/pretrained/baseline-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth")
        net_decoder = builder.build_decoder(
            arch="ppm_deepsup",
            fc_dim=2048,
            num_class=150,
            weights="./utils/pretrained/baseline-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth",
            use_softmax=True)
        crit = nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        self.segmentation_module.cuda()
        self.gpu = gpu
        self.cfg = cfg
        self.normalize = transforms.Normalize(
                mean=[102.9801, 115.9465, 122.7717],
                std=[1., 1., 1.])

    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def img_transform(self,img):
        # image to float
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img
        
    def preprocessImg(self, img=None):
        ori_height, ori_width, _ = img.shape
        img_resized_list = []
        for this_short_size in self.cfg.DATASET.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.cfg.DATASET.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = self.round2nearest_multiple(target_height, self.cfg.DATASET.padding_constant)
            target_width = self.round2nearest_multiple(target_width, self.cfg.DATASET.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)
        output = dict()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['img_ori'] = img
        return output

    def inference(self, img):
        self.segmentation_module.eval()
        inputs = self.preprocessImg(img)
        segSize = (inputs['img_ori'].shape[0],
                inputs['img_ori'].shape[1])
        img_resized_list = inputs['img_data']
        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, self.gpu)

            for img in img_resized_list:
                feed_dict = inputs.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                feed_dict = async_copy_to(feed_dict, self.gpu)

                # forward pass
                pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
        return pred
            
"""
cfg.merge_from_file("./config/ade20k-resnet50dilated-ppm_deepsup.yaml")
cm = calmask(cfg,gpu=0)
img = cv2.imread("./li.jpg", cv2.IMREAD_COLOR)
mask = cm.inference(img=img)
_max = pd.value_counts(mask.flatten()).keys()[0]
mask = np.where(mask == _max, 255, 0)
mask = Image.fromarray(mask.astype(np.uint8)).convert('L')
mask.save('./mask.png')
"""