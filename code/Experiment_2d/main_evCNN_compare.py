# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:33:47 2023

@author: andres cremades botella

File for evaluating the results of the CNN
"""
import ann_config as ann
import get_data_fun as gd


start = 1
step = 1
0   
dy = 1
dx = 1
CNN = ann.convolutional_residual()
CNN.load_model()
CNN.pred_rms_xy(testcases=True,step=step,down_y=dy,down_x=dx)
normdata = gd.get_data_norm()
normdata.geom_param(start,dy,dx)
normdata.read_Urms()
normdata.read_Urms_point()
CNN.plotrms_sim_xy_compare(normdata,xplus=4000,literature=True,padpix=15)