import time
import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F
import sys
from models.network import net

    
def load_model(device):
    print('Loading matting model ')
    myModel = net()
    myModel.eval()
    nm = {}
    state = torch.load('./pretrained/model.pth')
    for k in state.keys():
        if 'tracked' not in k:
            nm[k] = state[k]
    myModel.load_state_dict(nm)
    myModel.to(device)
    return myModel

class mat():
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        if use_gpu:
            self.device = torch.device('cuda:0')            
        else:
            self.device = torch.device('cpu')
        self.matmodel = load_model(self.device )

    def mat_processing(self,image,size,threshold):
        origin_h, origin_w, c = image.shape
        image_resize = cv2.resize(image, (size,size), interpolation=cv2.INTER_CUBIC)
        image_resize = (image_resize - (104., 112., 121.,)) / 255.0
        tensor_4D = torch.FloatTensor(1, 3, size, size)
        
        tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
        inputs = tensor_4D.to(self.device)

        trimap, alpha = self.matmodel(inputs)

        if not self.use_gpu:
            alpha_np = alpha[0,0,:,:].data.numpy()
        else:
            alpha_np = alpha[0,0,:,:].cpu().data.numpy()
        mask = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
        mask[mask<mask.max()*threshold] = mask.min()
        mask = (mask-mask.min())/(mask.max()-mask.min())*255
        mask = mask.astype(np.uint8)
        return mask
        
        """
        alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
        fg = np.multiply(alpha_np[..., np.newaxis], image)
        cv2.imwrite('./alpha.png',fg)
        bg = image
        bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
        bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

        bg[:,:,0] = bg_gray
        bg[:,:,1] = bg_gray
        bg[:,:,2] = bg_gray
        cv2.imwrite('./bg.png',bg)

        # fg[fg<=0] = 0
        # fg[fg>255] = 255
        # fg = fg.astype(np.uint8)
        # out = cv2.addWeighted(fg, 0.7, bg, 0.3, 0)
        out = fg + bg
        out[out<0] = 0
        out[out>255] = 255
        out = out.astype(np.uint8)

        return out
        """
"""
def main():
    img = cv2.imread("./imgs/li.jpg")
    cm = mat(use_gpu=True)
    mask = cm.mat_processing(img,1024)
    mask[mask > 0.2] = mask.max()
    mask = (mask-mask.min())/(mask.max()-mask.min())*255
    mask = mask.astype(np.uint8)
#    fg = np.multiply(mask[..., np.newaxis], img)
    cv2.imwrite('./alpha1.png',mask)


if __name__ == "__main__":
    main()

"""
