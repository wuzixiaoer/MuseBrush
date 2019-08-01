from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
import config
from utils.config import cfg
import json
import torch
from utils.stylizer import styleTrans,test_transform
from utils.genMask import calmask
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
from PIL import Image,ImageFilter

from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/')
def index():
    return redirect(url_for('go_into_a_painting'))

@app.route('/go_into_a_painting', methods=['POST', 'GET'])  # 添加路由
def go_into_a_painting():
    if request.method == 'POST':
        f = request.files['content']
        print(f.filename)
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        #user_input = request.form.get("name")

        basepath = os.path.abspath(os.path.dirname(__file__))  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'image1.jpg'), img)

        # cal mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        content = "static/images/image1.jpg"
        content = Image.open(content)

        cm = calmask(cfg,gpu=0)
        img = cv2.cvtColor(np.asarray(content), cv2.COLOR_RGB2BGR)

        mask = cm.inference(img=img)
        # _max = pd.value_counts(mask.flatten()).keys()[0]
        _mask = mask = np.where(mask == 12, 255, 0)
        mask = Image.fromarray(mask.astype(np.uint8)).convert('L')
        mask.save('static/mask.png')

        mask = mask.convert("RGBA")
        pixdata = mask.load()
        L, H = mask.size
        for l in range(L):
            for h in range(H):
                if pixdata[l, h][0] == 0 and pixdata[l, h][1] == 0 and pixdata[l, h][2] == 0:
                    pixdata[l, h] = (0, 0, 0, 0)

        mask.save('static/mask_new.png')


        imagenew = Image.new("RGBA", (512, 512))
        imagenew.paste(content,(0,0), mask=mask)
        imagenew.save('static/segmention.png')

        redirect('./style'))


    return render_template('upload.html')

@app.route('/style', methods=['POST','GET'])
def style():
    if request.method == 'POST':
        style_label = int(request.form.get('style'))
        style = Image.open("image"+style_label+".png")
        # 进行风格迁移
        content = Image.open(content)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        styleTransfer(content,style,device)
    return render_template('style.html')


def styleTransfer(content,device):
    vgg_path='./pretrained/style_models/vgg_normalised.pth'
    decoder_path='./pretrained/style_models/decoder_iter_76000.pth'
    transform_path='./pretrained/style_models/sa_module_iter_76000.pth'
    crop='store_true'
    content_size=512
    style_size=512
    alpha=0.6
    content_tf = test_transform(content_size,crop)
    style_tf = test_transform(style_size,crop)
    _content = content_tf(content)
    _style = style_tf(style)

    _style = _style.to(device).unsqueeze(0)
    _content = _content.to(device).unsqueeze(0)
    transformer = styleTrans(device=device,vgg_path=vgg_path,
                            transform_path=transform_path,
                            decoder_path=decoder_path)
    with torch.no_grad():
        content_trans = transformer.stansform(content=_content,style=_style,alpha=alpha)



@app.route('/result', methods=['POST', 'GET'])
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(host="localhost",port=8080,debug=True)
