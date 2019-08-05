import _init_paths
from utils.transfer import style_transfer

t = style_transfer()
style_dict = {'style_src':'./utils/imgs/3.jpg', 'patch_src':'./utils/imgs/3_patch.jpg',
         'loc': (420, 250), 'alpha':0.9, 'gl_ratio':0.9, 'hsize': 128}
result = t.transfer('./utils/imgs/girl.jpg', style_dict)
# 3.jpg 420,250 128