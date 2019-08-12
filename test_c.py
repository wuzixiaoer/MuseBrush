import _init_paths
from PIL import Image
from utils.transfer import style_transfer

t = style_transfer()
style_dict = {'style_src':'./utils/imgs/shouter.jpg', 'patch_src':'./utils/imgs/shouter_patch.jpg',
         'loc': (50, 500), 'alpha':0.9, 'gl_ratio':0.1, 'hsize':300, 'bg':None}
result = t.transfer(Image.open('./utils/imgs/brad_pitt.jpg'), style_dict)
result.save('utils/result/result.png', quality=95)
# coast.jpg 700,420 0.9 0.8 128
# 4.jpg 1300, 600 0.9 0.8 256
# Edouard 280, 450 0.9 0.8 750
# Egon 400, 0 0.9 0.2 818
# timg 445, 240 0.9 0.1 260 big head
# sunday 900, 650 0.8 0.2 400
# Miklmaid 1050, 1300 0.9 0.2 300
# reaper 1150, 1000 0.9 0.6 400
# chamber 200, 200 0.9, 0.6 400
# woman 0,0 0.9 0.2 1132
# shouter 80, 500 0.9 0.1 300