# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:33:47 2023

@author: andres cremades botella

File for evaluating the results of the CNN
"""
import ann_config as ann
import get_data_fun as gd

dy = 1
dx = 1
CNN = ann.convolutional_residual() 
CNN.load_model()
normdata = gd.get_data_norm()
normdata.geom_param(1,delta_x=dx,delta_y=dy)
CNN.plot_flowfield(normdata,1,facerr=0.1)
CNN.mre_pred(normdata,testcases=True,step=10)
CNN.savemre()