# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 02:58:15 2017

@author: Khalil
"""

import glob
from PIL import Image

GIF_folder = 'progress'
# filepaths
fp_in = "./%s/tradespace_*.png" %(GIF_folder)
fp_out = "./%s/tradespace.gif" %(GIF_folder)

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)