from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
import cv2
import argparse
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import net
from stylizer import styleTrans,test_transform


vgg_path='./models/vgg_normalised.pth'
alpha=0.7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask = "./mask_cat.jpg"
style = "./target_cat.jpg"
style = Image.open(style)
tf = test_transform((675,1200), False)
style = tf(style)
mask =  cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

decoder_path='./experiments/decoder_iter_92000.pth'
transform_path='./experiments/sa_module_iter_92000.pth'
transformer = styleTrans(device=device,vgg_path=vgg_path,
                        transform_path=transform_path,
                        decoder_path=decoder_path,)

# 修改这里处理视频
def transfer(frame,style):
    content = tf(frame)
    style = style.to(device).unsqueeze(0)
    content = content.to(device,dtype=torch.float32).unsqueeze(0)
    print(content.shape)
    print(style.shape)
    output = transformer.stansform(content=content,style=style, mask=mask, alpha=alpha)
    output = output.cpu()
    del style
    del content
    return output

imgspath = './imgs/'
imgsrespath = './imgsres/'
if not os.path.exists(imgspath):
    os.makedirs(imgspath)
if not os.path.exists(imgsrespath):
    os.makedirs(imgsrespath)

"""step1： 读取视频"""
print('step1： 读取视频...')
vc = cv2.VideoCapture("./catdemo.mp4")
c = 1
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
while rval:
    rval, frame = vc.read()
    cv2.imwrite(imgspath + str(c) + '.jpg', frame)  # 需要先存在这个文件夹
    c = c + 1
    cv2.waitKey(1)
vc.release()
print('Done')


"""step2: 处理帧"""
print("step2: 风格迁移...")
im_names = os.listdir(imgspath)
for i in range(len(im_names)-2):
    imgname = imgspath + str(i+1) + '.jpg'
    ori_img = cv2.imread(imgname)
    ori_img = Image.fromarray(ori_img.astype('uint8')[:, :, ::-1], mode='RGB')
    frame = transfer(ori_img,style)
    save_image(frame, imgsrespath + str(i+1) + '.jpg')
    del frame

print('Done')


"""step3: 回传视频"""
print('step3: 回传视频...')
# Edit each frame's appearing time!
fps = 25
fourcc = VideoWriter_fourcc(*"MJPG")
# 分辨率自己看着写
videoWriter = cv2.VideoWriter("pikaroom2Res.avi", fourcc, fps, (1200, 680))

im_names = os.listdir(imgsrespath)
for im_name in range(len(im_names)-2):
    frame = cv2.imread(imgsrespath + str(im_name+1) + '.jpg')
    print(im_name)
    videoWriter.write(frame)

videoWriter.release()
print('All Done! Congratulation!')