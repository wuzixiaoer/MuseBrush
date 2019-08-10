import _init_paths
from PIL import Image
from utils.transfer import style_transfer

t = style_transfer()
style_dict = {'style_src':'./utils/imgs/Arles.jpg', 'patch_src':'./utils/imgs/Arles_patch.jpg',
         'loc': (230, 600), 'alpha':0.9, 'gl_ratio':0.7, 'hsize': 196}
result = t.transfer(Image.open('./utils/imgs/lty.jpg'), style_dict)
result.save('utils/result/result.png', quality=95)
# coast.jpg 700,420 0.9 0.8 128
# 4.jpg 1300, 600 0.9 0.8 256
# Edouard 280, 450 0.9 0.8 750
# Egon 400, 0 0.9 0.2 818
# timg 445, 240 0.9 0.1 260 big head
# sunday 900, 650 0.8 0.2 400
# Miklmaid 1050, 1300 0.9 0.2 300
# reaper 1150, 1000 0.9 0.6 400
