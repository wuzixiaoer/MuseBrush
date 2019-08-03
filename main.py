import os
import cv2
import time
import json
import torch
import config
import sys
sys.path.append('./utils/')

import numpy as np
import pandas as pd
from datetime import timedelta
from werkzeug.utils import secure_filename

from utils.config import cfg
from utils.matting import mat
from utils.stylizer import styleTrans,test_transform

from torchvision.utils import make_grid
from PIL import Image,ImageFilter

from flask import Flask, Blueprint, render_template, request, redirect, url_for, make_response, jsonify


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

class PrefixMiddleware(object):
    def __init__(self, app, prefix='/infer-8438a117-fbef-4184-a6e2-c6ed2d7b224f'):
        self.app=app
        self.prefix=prefix
    def __call__(self, environ, start_response):
        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO']=environ['PATH_INFO'][len(self.prefix):]
            environ['SCRIPT_NAME']=self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ['This url does not belong to the app.'.encode()]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#bp = Blueprint('burritos', __name__, url_prefix='/infer-8438a117-fbef-4184-a6e2-c6ed2d7b224f')
app = Flask(__name__)
#app.register_blueprint(bp, url_prefix='/infer-8438a117-fbef-4184-a6e2-c6ed2d7b224f')
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/infer-8438a117-fbef-4184-a6e2-c6ed2d7b224f')
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
        content = "static/images/image1.jpg"
        content = Image.open(content)

        cm = mat(use_gpu = True)
        img = cv2.cvtColor(np.asarray(content),cv2.COLOR_RGB2BGR)  

        mask = cm.mat_processing(img, 512, 0.9)
        mask = Image.fromarray(mask).convert('L')
        mask.save('static/mask.png')
        im = Image.new("RGB", mask.size)
        im.paste(content, mask=mask)
        im.save('static/mask_content.png')

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

        return redirect('/style')


    return render_template('upload.html')

@app.route('/style', methods=['POST','GET'])
def style():
    if request.method == 'POST':

        style_label = request.form.get('style')

        print(style_label)

        style = Image.open("static/style_img/"+style_label+".jpg")
        # 进行风格迁移
        content = "static/images/image1.jpg"
        content = Image.open(content)
        
        content_s = styleTransfer(content,style,device) # tensor

        grid = make_grid(content_s, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
        output_ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        content_s = Image.fromarray(output_ndarr)

        content_s.save('static/content_transfered.png')

    return render_template('style.html')


def styleTransfer(content, style, device):
    vgg_path='./utils/pretrained/style_models/vgg_normalised.pth'
    decoder_path='./utils/pretrained/style_models/decoder_iter_100000.pth'
    transform_path='./utils/pretrained/style_models/sa_module_iter_100000.pth'
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

    return content_trans


@app.route('/result', methods=['POST', 'GET'])
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(host="localhost",port=8080,debug=True,threaded=True)

