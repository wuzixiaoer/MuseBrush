from transfer import style_transfer

t = style_transfer()
style_dict = {'style_src':'./imgs/3.jpg', 'patch_src':'./imgs/3_patch.jpg',
         'loc': (0,0), 'alpha':0.8, 'gl_ratio':0.8}
result = t.transfer('./imgs/girl.jpg', style_dict)