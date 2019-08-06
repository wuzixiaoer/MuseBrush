import _init_paths
from utils.transfer import style_transfer

t = style_transfer()
style_dict = {'style_src':'./utils/imgs/beach.jpg', 'patch_src':'./utils/imgs/beach.jpg',
         'loc': (600, 1400), 'alpha':0.9, 'gl_ratio':0.2, 'hsize': 400}
result = t.transfer('./utils/imgs/test.jpg', style_dict)
# 3.jpg 420,250 128
# 4.jpg 1300, 600 0.9 0.8 256
# Edouard 350, 650 0.9 0.2 1800
# Egon 400, 0 0.9 0.2 818