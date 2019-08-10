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
    ['Landscape', (310, 440), 0.9, 0.6, 128],
    ['Twodogs', (485, 480), 0.95, 0.5, 176],
    ['Arles', (230, 600), 0.9, 0.7, 196],
    ['coast', (700, 420), 0.9, 0.8, 128],
    ['sunday', (900, 650), 0.8, 0.2, 400],
    ['Milkmaid', (1050, 1300), 0.9, 0.2, 300],
    ['reaper', (1150, 1000), 0.9, 0.6, 400],
    ['chamber', (200, 200), 0.9, 0.6, 400],
    ['woman', (0, 0), 0.9, 0.2, 1132],
    ['van', (0, 0), 0.7, 0.2, 778],
    ['BertheMorisot', (200, 300), 0.9, 0.2, 1046]
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
                  'loc': styles[style_id][1], 'alpha':styles[style_id][2], 'gl_ratio':styles[style_id][3], 'hsize': styles[style_id][4]}
    result = model.transfer(content, style_dict)
    print('get result')
    img_buffer = BytesIO()
    result.save(img_buffer, 'jpeg')
    base64_str = base64.b64encode(img_buffer.getvalue())
    print('Done')
    return base64_str

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)

