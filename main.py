import os
from io import BytesIO
import base64
from PIL import Image

import numpy as np
# from datetime import timedelta

import _init_paths
from flask import Flask, send_file, request
from utils.transfer import style_transfer

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


app = Flask(__name__)
app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/infer-8438a117-fbef-4184-a6e2-c6ed2d7b224f')


styles = [
    ['Landscape', (400, 572), 0.9, 0.6, 250, 0],
    ['Twodogs', (500, 749), 0.9, 0.6, 300, 0],
    ['Arles', (300, 650), 0.9, 0.7, 300, 0],
    ['coast', (370, 300), 0.9, 0.5, 200, 0],
    ['sunday', (1100, 860), 0.8, 0.2, 380, 0],
    ['Milkmaid', (650, 1066), 1, 0.2, 332, 0],
    ['reaper', (1250, 1419), 0.9, 0.1, 400, 0],
    ['chamber', (300, 525), 0.8, 0.6, 350, 0],
    ['woman', (512, 716), 0.7, 0.2, 832, 1],
    ['shouter', (150, 650), 0.9, 0.1, 300, 0],
    ['BertheMorisot', (480, 923), 0.7, 0.1, 846, 1]
]

model = style_transfer()
@app.route('/style', methods=['POST', 'GET'])
def sf():
    print(request.method)
    print(request.form.to_dict())
    content = Image.open(request.files['content'])
    print(type(content))
    style_id = int(request.form['style_id'])
    print(style_id)
    style_dict = {'style_src':os.path.join('utils/imgs/',styles[style_id][0]+'.jpg'), 'patch_src':os.path.join('utils/imgs/', styles[style_id][0]+'_patch.jpg'),
                  'loc': styles[style_id][1], 'alpha':styles[style_id][2], 'gl_ratio':styles[style_id][3], 'hsize': styles[style_id][4], 'bg': os.path.join('utils/imgs/',styles[style_id][0]+'_bg.jpg') if styles[style_id][-1]==1 else None}
    result = model.transfer(content, style_dict)
    print('get result')
    img_buffer = BytesIO()
    result.save(img_buffer, 'jpeg')
    base64_str = base64.b64encode(img_buffer.getvalue())
    print('Done')
    return base64_str

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)

