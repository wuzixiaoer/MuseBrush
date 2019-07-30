from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
import urllib
import re

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
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        # img = cv2.imread(upload_path)
        # cv2.imwrite(os.path.join(basepath, 'static/images', 'content.jpg'), img)

        return redirect(url_for('style'))

    return render_template('upload.html')


@app.route('/style', methods=['GET', 'POST'])
def style():
    
    path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    stylepath = os.path.join(path, 'static/i')
    img = cv2.imread(stylepath)
    cv2.imwrite(os.path.join(path, 'static/images', 'style.jpg'), img)

    return render_template('style.html')



@app.route('/result', methods=['POST', 'GET'])
def result():
    return render_template('result.html')

    