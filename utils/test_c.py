from transfer import style_transfer

t = style_transfer()
style_dict = {'style_src':'./imgs/3.jpg', 'patch_src':'./imgs/3_patch.jpg',
         'loc': (420, 250), 'alpha':0.9, 'gl_ratio':0.9, 'hsize': 128}
result = t.transfer('./imgs/girl.jpg', style_dict)
# 3.jpg 420,250 128