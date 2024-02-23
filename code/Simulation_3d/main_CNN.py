# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:10:03 2023

@author: andres cremades botella

File containing the configuration of the CNN model and the training process
"""
import ann_config as ann
import numpy as np
import get_data_fun as gd


CNN = ann.convolutional_residual(ngpu=1)
dy = 1
dz = 1
dx = 1
shpy = int((201-1)/dy)+1
shpz = int((96-1)/dz)+1
shpx = int((192-1)/dx)+1
CNN.define_model(shp=(shpy,shpz,shpx,3),learat=1e-2,nfil=np.array([4,8,16])) 
CNN.train_model(7000,7010,delta_t=10,delta_e=20,max_epoch=2e1,\
                batch_size=4,down_y=dy,down_z=dz,down_x=dx)

print('End: main_CNN.py')