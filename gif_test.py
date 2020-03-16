# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 02:58:15 2017

@author: Khalil
"""

import os, imageio

frame_dumpfile = 'frame_names_Magnitude_Detail.pkl'

def create_gif( filenames,duration,current_path,GIF_folder ):
    images = []
    for filename in filenames:
        print(filename)
        filename = os.path.join(current_path,GIF_folder,filename)
        img = imageio.imread(filename)
#        img = imageio.imwrite(filename,img,optimize=False)
        images.append(img)
    output_file = '%s.gif' %(GIF_folder)
    output_file = os.path.join(current_path,GIF_folder,output_file)
    imageio.mimsave(output_file, images, duration=duration)

current_path = os.getcwd() # Working directory of file
GIF_folder = 'progress'

filenames = [];
for i in range(59):
    filenames += ['tradespace_%i.png' %(i+1)]

duration = 0.5
create_gif(filenames, duration,current_path,GIF_folder)