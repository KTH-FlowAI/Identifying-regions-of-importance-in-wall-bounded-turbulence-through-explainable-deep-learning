# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:33:47 2023

@author: andres cremades botella

File for evaluating the results of the CNN
"""
import ann_config as ann
import get_data_fun as gd

start = 7000 
end = 7010
step = 1 
dy = 1
dz = 1
dx = 1
CNN = ann.convolutional_residual() 
CNN.load_model()
normdata = gd.get_data_norm()
normdata.geom_param(start,dy,dz,dx)
CNN.plot_flowfield(normdata,start,12,facerr=0.1)
CNN.mre_pred(normdata,start,end,step)
CNN.savemre()