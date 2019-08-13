import _init_paths
from PIL import Image
from utils.transfer import style_transfer

t = style_transfer()
style_dict = {'style_src':'./utils/imgs/BertheMorisot.jpg', 'patch_src':'./utils/imgs/BertheMorisot_patch.jpg',
         'loc': (480, 923), 'alpha':0.8, 'gl_ratio':0.1, 'hsize':846, 'bg':'./utils/imgs/BertheMorisot_bg.jpg'}
result = t.transfer(Image.open('./utils/imgs/hjy.jpg'), style_dict)
result.save('utils/result/result.png', quality=95)
# coast.jpg 800,508 0.9 0.2 150
# Miklmaid 1150, 1577 1 0.9 300
# reaper 1250, 1419 0.9 0.1 400
# chamber 300, 525 0.8, 0.6 350
# woman 512,716 0.9 0.2 832
# shouter 150, 650 0.9 0.1 300
# Landscape 400, 572, 0.9, 0.6, 250
# Twodogs 500, 749 0.9 0.6 300 0
# Arles 300, 650 0.9 0.7 300
# sunday 1100 860 0.8 0.1 380
# BertheMorisot 480, 923 0.8 0.1 846

