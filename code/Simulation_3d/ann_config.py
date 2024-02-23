# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:12:37 2023

@author: andres cremades botella

file containing the functions to configurate the CNN
"""
import numpy as np
import os

def nearest(array,value):
    array_value = abs(array-value)
    index = np.argmin(array_value)
    nearest = array[index]
    return nearest,index

def plottrain(file="../../results/Simulation_3d/hist.txt"):
    """
    Function for plotting the training of the neural network
    """ 
    with open(file, 'r') as fread:
        data_train = np.array([[float(ii) for ii in line.split(',')] \
                                for line in fread])
    import matplotlib.pyplot as plt
    fs = 16
    plt.figure()
    plt.plot(data_train[:,0],data_train[:,2], color='#7AD151',\
             label='Validation loss',linewidth=2)
    plt.plot(data_train[:,0],data_train[:,1], color='#440154',\
             label='Training loss',linewidth=2)
    plt.title('Training and validation loss',fontsize=fs)
    plt.xlabel('Epoch',fontsize=fs)
    plt.ylabel('Loss function (-)',fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)  
    plt.yscale('log')
    plt.legend(fontsize=fs)
    plt.grid()
    plt.tight_layout()
    plt.savefig('../../results/Simulation_3d/Loss_plot.png')
    plt.show()


def block(xx,nfil,stride,activ,kernel):
    """
    Function for configuring the CNN block
    xx     : input data
    nfil   : number of filters of the channels
    stride : size of the strides
    activ  : activation function
    kernel : size of the kernel
    -----------------------------------------------------------------------
    xx     : output data
    """
    from tensorflow.keras.layers import Conv3D, BatchNormalization,\
        Activation
    xx = Conv3D(nfil, kernel_size=kernel, 
                strides=(stride,stride,stride),padding="same")(xx)
    xx = BatchNormalization()(xx) 
    xx = Activation(activ)(xx)
    return xx


def lastblock(xx,nfil,stride,kernel):
    """
    Function for configuring the last CNN block of a residual loop
    xx     : input data
    nfil   : number of filters of the channels
    stride : size of the strides
    activ  : activation function
    kernel : size of the kernel
    -----------------------------------------------------------------------
    xx     : output data
    """
    from tensorflow.keras.layers import Conv3D, BatchNormalization,\
        Activation
    xx = Conv3D(nfil, kernel_size=kernel, 
                strides=(stride,stride,stride),padding="same")(xx)
    xx = BatchNormalization()(xx) 
    return xx



def invblock(xx,nfil,stride,activ,kernel,outpad=(1,1,1)):
    """
    Function for configuring the inverse CNN block
    xx     : input data
    nfil   : number of filters of the channels
    stride : size of the strides
    activ  : activation function
    kernel : size of the kernel
    -----------------------------------------------------------------------
    xx     : output data
    """
    from tensorflow.keras.layers import Conv3DTranspose, BatchNormalization,\
        Activation 
    xx = Conv3DTranspose(nfil, kernel_size=kernel,
                         strides=(stride,stride,stride),padding="valid",output_padding=outpad)(xx)
    xx = BatchNormalization()(xx) 
    xx = Activation(activ)(xx)
    return xx


def residual_block(xx,nfil,stride,activ,kernel):
    """
    Function for configuring the CNN block
    xx     : input data
    nfil   : number of filters of the channels
    stride : size of the strides
    activ  : activation function
    kernel : size of the kernel
    -----------------------------------------------------------------------
    out     : output data
    """
    from tensorflow.keras.layers import Conv3D, BatchNormalization,\
        Activation, Add
    fx = block(xx,nfil,stride,activ,kernel)
    fx = lastblock(fx,nfil,stride,kernel)
    out = Add()([xx, fx])
    out = Activation(activ)(out)
    return out


class convolutional_residual():
    """
    Class for creating a convolutional neural network with a residual layer
    """
    def __init__(self,ngpu=1,fileddbb='../../data/P125_21pi_vu/P125_21pi_vu',\
                 pond='none'):
        self.devices(ngpu)
        self.fileddbb = fileddbb
        self.pond = pond
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass
                
    def devices(self,ngpu):
        """
        Create the list of gpu
        """
        dev_list = str(np.arange(ngpu).tolist())
        self.cudadevice = dev_list.replace('[','').replace(']','')
    
    
    def model_base(self,shp,nfil,stride,activ,kernel,padpix):
        """
        Base structure of the model, with residual blocks
        attached.
            shp     : input shape
            nfil    : number of filters, each item is the value of a layer
            stride  : value of the stride in the 3 dimensions of each layer
            activ   : activation functions of each layer
            kernel  : size of the kernel in each layer
            -------------------------------------------------------------------
            inputs  : input data of the model
            outputs : output data of the model
        """
        from tensorflow.keras.layers import Input,MaxPool3D,\
        Concatenate,Add,Activation
        from tensorflow.image import crop_to_bounding_box
        dim0 = shp[0]
        dim1 = shp[1]+2*padpix
        dim2 = shp[2]+2*padpix
        dim3 = shp[3]
        shp = (dim0,dim1,dim2,dim3)
        self.inputs = Input(shape=shp)
        # First layer
        xx11 = block(self.inputs,nfil[0],stride[0],activ[0],kernel[0]) 
        xx12 = block(xx11,nfil[0],stride[0],activ[0],kernel[0]) 
        xx13 = block(xx12,nfil[0],stride[0],activ[0],kernel[0])
        xx14 = block(xx13,nfil[0],stride[0],activ[0],kernel[0])
        # to second layer
        xx20 = MaxPool3D(3)(xx14)
        # second layer
        xx21 = block(xx20,nfil[1],stride[1],activ[1],kernel[1])
        xx22 = block(xx21,nfil[1],stride[1],activ[1],kernel[1])
        xx23 = block(xx22,nfil[1],stride[1],activ[1],kernel[1])
        xx24 = block(xx23,nfil[1],stride[1],activ[1],kernel[1])
        # to third layer
        xx30 = MaxPool3D(3)(xx24)
        # third layer
        xx31 = block(xx30,nfil[2],stride[2],activ[2],kernel[2])
        xx32 = block(xx31,nfil[2],stride[2],activ[2],kernel[2])
        xx33 = block(xx32,nfil[2],stride[2],activ[2],kernel[2])
        xx34 = block(xx33,nfil[2],stride[2],activ[2],kernel[2])
        # go to second layer
        xx20b = invblock(xx34,nfil[1],3,activ[1],kernel[1],outpad=(1,0,2))
        xx21b = Concatenate()([xx24,xx20b])   
        # second layer
        xx22b = block(xx21b,nfil[1],stride[1],activ[1],kernel[1])
        xx23b = block(xx22b,nfil[1],stride[1],activ[1],kernel[1])
        xx24b = block(xx23b,nfil[1],stride[1],activ[1],kernel[1])
#        # go to first layer
        xx10b = invblock(xx24b,nfil[0],3,activ[0],kernel[0],outpad=(0,0,0)) 
        xx11b = Concatenate()([xx10b,xx14])
        # First layer
        xx12b = block(xx11b,nfil[0],stride[0],activ[0],kernel[0])
        xx13b = block(xx12b,nfil[0],stride[0],activ[0],kernel[0])
        xx14b = block(xx13b,nfil[0],stride[0],activ[0],kernel[0])
        xx15b = block(xx14b,3,stride[0],activ[0],kernel[0])
        #
        xx16b = xx15b[:,:,padpix:-padpix,padpix:-padpix,:]
        self.outputs = xx16b
        
    
    def define_model(self,shp=(201,96,192,3),nfil=np.array([32,64,96]),\
                     stride=np.array([1,1,1]),\
                     activ=["relu","relu","relu"],\
                     kernel=[(3,3,3),(3,3,3),(3,3,3)],optmom=0.9,\
                     learat=0.001,padpix=15):
        """
        Define the model with the strategy
            shp     : input shape
            nfil    : number of filters, each item is the value of a layer
            stride  : value of the stride in the 3 dimensions of each layer
            activ   : activation functions of each layer
            kernel  : size of the kernel in each layer
            optmom  : optimizer momentum
            learat  : learning rate
        """
        import os
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.optimizers import RMSprop
        os.environ["CUDA_VISIBLE_DEVICES"]= self.cudadevice
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        physical_devices = tf.config.list_physical_devices('GPU')
        available_gpus   = len(physical_devices)
        print('Using TensorFlow version: ', tf.__version__, ', GPU:',available_gpus)
#        print('Using Keras version: ', tf.keras.__version__)
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError as ee:
                print(ee)
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        print("GPUs in use:")
        for gpu in tf.config.list_logical_devices('GPU'):
            print(gpu.name)
        with strategy.scope(): 
            self.model_base(shp,nfil,stride,activ,kernel,padpix)
            optimizer = RMSprop(learning_rate=learat,momentum=optmom) 
            self.model = Model(self.inputs, self.outputs)
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),\
                               optimizer=optimizer)
        self.model.summary()    
        self.options = tf.data.Options()
        self.options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.FILE
        
    def train_model(self,start,end,delta_t=10,delta_e=20,max_epoch=100,\
                    batch_size=1,down_y=1,down_z=1,down_x=1,\
                    trainfile='../../results/Simulation_3d/trained_model.h5',trainhist='../../results/Simulation_3d/hist.txt',\
                    delta_pred=1,padpix=15):
        """
        Function for training the CNN model
            start   : initial field index
            end     : final field index
            delta_t : time interval loaded
            delta_e : epoch in training
            max_epoch : maximum number of epoch in each package of flow fields
            batch_size: batch size
            down_y    : downsizing in y
            down_z    : downsizing in z
            down_x    : donwsizing in x
            fileddbb  : path to the ddbb files
            trainfile : file for saving the training
            trainhist : file for saving the loss evolution during training
        """        
        import get_data_fun as gd
        import pandas as pd
        # Create a vector with the random index of the training fields
        ind_vec = np.array(range(start,end-delta_pred))
        np.random.shuffle(ind_vec)
        # Divide the previous vector in smaller packages using the delta_t
        ii_ini = 0
        ii_fin = ii_ini+delta_t
        # Initialize and normalize the data
        data = gd.get_data_norm(self.fileddbb,pond=self.pond)
        data.geom_param(start,down_y,down_z,down_x)
        try:
            data.read_norm()
        except:
            data.calc_norm(start,end)
            data.save_norm()
        epochcum = 0
        while ii_ini < end-start:
            if ii_fin < end-start:
                interval = ind_vec[ii_ini:ii_fin]
            else:
                interval = ind_vec[ii_ini:]
            train_data,val_data = data.trainvali_data(interval,\
                                                      delta_pred=delta_pred,\
                                                      padpix=padpix)
            train_data = train_data.batch(batch_size)
            val_data = val_data.batch(batch_size) 
            train_data = train_data.with_options(self.options)
            val_data = val_data.with_options(self.options)
            epoch = 0
            while epoch < max_epoch: 
                print('Training... '+str(ii_ini/(end-start)*100)+'%')
                data_training = self.model.fit(train_data,\
                                               batch_size=batch_size,\
                                               verbose=2,epochs=delta_e,\
                                               validation_data=val_data)  
                hmat = np.zeros((delta_e,3))
                if ii_ini == 0 and epoch == 0:
                    hmat[:,0] = np.arange(delta_e)
                    hmat[:,1] = data_training.history['loss']
                    hmat[:,2] = data_training.history['val_loss']
                    with open(trainhist,'w') as filehist:
                        for line in hmat:
                            filehist.write(str(line[0])+','+str(line[1])+','+\
                                           str(line[2])+'\n')
                else:
                    print(epoch)
                    hmat[:,0] = np.arange(delta_e)+epochcum
                    hmat[:,1] = data_training.history['loss']
                    hmat[:,2] = data_training.history['val_loss']
                    with open(trainhist,'a') as filehist:
                        for line in hmat:
                            filehist.write(str(line[0])+','+str(line[1])+','+\
                                           str(line[2])+'\n')                
                self.model.save(trainfile)
                epoch += delta_e
                epochcum += delta_e
            ii_ini = ii_fin
            ii_fin = ii_ini+delta_t
                        
    def load_ANN(self,filename='../../results/Simulation_3d/trained_model.h5'):
        """ 
        Function for loading the ANN model
        """
        import tensorflow as tf 
        self.model = tf.keras.models.load_model(filename)
    
    def load_model(self,filename='../../results/Simulation_3d/trained_model.h5'): 
        """
        Function for loading the tensorflow model for training
            * filename : file name of the model
        """
        import os
        import tensorflow as tf
        os.environ["CUDA_VISIBLE_DEVICES"]= self.cudadevice
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        physical_devices = tf.config.list_physical_devices('GPU')
        available_gpus   = len(physical_devices)
        print('Using TensorFlow version: ', tf.__version__, ', GPU:',available_gpus)
#        print('Using Keras version: ', tf.keras.__version__)
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError as ee:
                print(ee)
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        print("GPUs in use:")
        for gpu in tf.config.list_logical_devices('GPU'):
            print(gpu.name)
        with strategy.scope(): 
            self.model = tf.keras.models.load_model(filename)
        self.model.summary()    
        self.options = tf.data.Options()
        self.options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.FILE
        
    def eval_model(self,index,down_y=1,down_z=1,down_x=1,start=1000,end=7000,\
                   padpix=15):
        """
        Function for evaluating the model
            * index : index of the file
        """
        import get_data_fun as gd
        data = gd.get_data_norm(self.fileddbb,pond=self.pond)
        data.geom_param(start,down_y,down_z,down_x)
        try:
            data.read_norm()
        except:
            data.calc_norm(start,end)
        try:
            data.UUmean 
        except:
            data.read_Umean()
        uu,vv,ww = data.read_velocity(index,padpix=padpix)
        past_field = data.norm_velocity(uu,vv,ww,padpix=padpix)
        pred_field = self.model.predict(past_field)
        pred_field_dim = data.dimensional_velocity(pred_field)
        past_field_dim = np.zeros((data.my,data.mz+2*padpix,data.mx+2*padpix,3))
        past_field_dim[:,:,:,0] = uu
        past_field_dim[:,:,:,1] = vv
        past_field_dim[:,:,:,2] = ww
        self.my = data.my
        self.mz = data.mz
        self.mx = data.mx
        return pred_field_dim
        
    def _calc_rms(self,uu,vv,ww,uu2_cum=0,vv2_cum=0,ww2_cum=0,uv_cum=0,\
              vw_cum=0,uw_cum=0,nn_cum=0):
        uu2 = np.multiply(uu,uu)
        vv2 = np.multiply(vv,vv)
        ww2 = np.multiply(ww,ww)
        uv  = np.multiply(uu,vv)
        vw  = np.multiply(vv,ww)
        uw  = np.multiply(uu,ww)
        uu2_cum += np.sum(uu2,axis=(1,2))
        vv2_cum += np.sum(vv2,axis=(1,2))
        ww2_cum += np.sum(ww2,axis=(1,2))
        uv_cum  += np.sum(uv,axis=(1,2))
        vw_cum  += np.sum(vw,axis=(1,2))
        uw_cum  += np.sum(uw,axis=(1,2))
        nn_cum += np.ones((self.my,))*self.mx*self.mz
        return uu2_cum,vv2_cum,ww2_cum,uv_cum,vw_cum,uw_cum,nn_cum
    
    
    def pred_rms(self,start,end,step=1,down_y=1,down_z=1,down_x=1,padpix=15):
        """
        Function for calculating the rms of the velocity components and the 
        product of the velocity fluctuations of the predicted fields
        """
        for ii in range(start,end,step):
            pfield = self.eval_model(ii,down_y=down_y,down_z=down_z,\
                                     down_x=down_x,start=start,padpix=padpix)
            uu  = pfield[:,:,:,0]
            vv  = pfield[:,:,:,1]
            ww  = pfield[:,:,:,2]
            if ii==start:
                uu2_cum,vv2_cum,ww2_cum,uv_cum,vw_cum,uw_cum,nn_cum =\
                self._calc_rms(uu,vv,ww)
            else:
                uu2_cum,vv2_cum,ww2_cum,uv_cum,vw_cum,uw_cum,nn_cum =\
                self._calc_rms(uu,vv,ww,uu2_cum=uu2_cum,vv2_cum=vv2_cum,\
                               ww2_cum=ww2_cum,uv_cum=uv_cum,vw_cum=vw_cum,\
                               uw_cum=uw_cum,nn_cum=nn_cum)
        self.uurms = np.sqrt(np.divide(uu2_cum,nn_cum))    
        self.vvrms = np.sqrt(np.divide(vv2_cum,nn_cum))   
        self.wwrms = np.sqrt(np.divide(ww2_cum,nn_cum)) 
        self.uv    = np.divide(uv_cum,nn_cum)
        self.vw    = np.divide(vw_cum,nn_cum)
        self.uw    = np.divide(uw_cum,nn_cum)
            
        
            
    def saverms(self,file="../../results/Simulation_3d/Urmspred.txt"):
        """
        Function for saving the value of the rms velocity
        """
        file_save = open(file, "w+")           
        content = str(self.uurms.tolist())+'\n'
        file_save.write(content)    
        content = str(self.vvrms.tolist())+'\n'
        file_save.write(content)    
        content = str(self.wwrms.tolist())+'\n'
        file_save.write(content)          
        content = str(self.uv.tolist())+'\n'
        file_save.write(content)    
        content = str(self.vw.tolist())+'\n'
        file_save.write(content)    
        content = str(self.uw.tolist())+'\n'
        file_save.write(content) 
        
    def readrms(self,file="../../results/Simulation_3d/Urmspred.txt"):
        """
        Read the predicted rms velocity
        """
        file_read = open(file,"r")
        self.uurms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvrms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.wwrms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uv = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vw = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uw = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        
    def plotrms(self,data):
        """
        Function for plotting the rms
        """        
        uurms_dplus_data = data.uurms[:data.yd_s]/data.vtau
        vvrms_dplus_data = data.vvrms[:data.yd_s]/data.vtau
        wwrms_dplus_data = data.wwrms[:data.yd_s]/data.vtau
        uurms_uplus_data = np.flip(data.uurms[data.yu_s:])/data.vtau
        vvrms_uplus_data = np.flip(data.vvrms[data.yu_s:])/data.vtau
        wwrms_uplus_data = np.flip(data.wwrms[data.yu_s:])/data.vtau
        uv_dplus_data    = data.uv[:data.yd_s]/data.vtau**2
        vw_dplus_data    = data.vw[:data.yd_s]/data.vtau**2
        uw_dplus_data    = data.uw[:data.yd_s]/data.vtau**2
        uv_uplus_data    = -np.flip(data.uv[data.yu_s:])/data.vtau**2
        vw_uplus_data    = -np.flip(data.vw[data.yu_s:])/data.vtau**2
        uw_uplus_data    = np.flip(data.uw[data.yu_s:])/data.vtau**2
        uurms_dplus_pred = self.uurms[:data.yd_s]/data.vtau
        vvrms_dplus_pred = self.vvrms[:data.yd_s]/data.vtau
        wwrms_dplus_pred = self.wwrms[:data.yd_s]/data.vtau
        uurms_uplus_pred = np.flip(self.uurms[data.yu_s:])/data.vtau
        vvrms_uplus_pred = np.flip(self.vvrms[data.yu_s:])/data.vtau
        wwrms_uplus_pred = np.flip(self.wwrms[data.yu_s:])/data.vtau
        uv_dplus_pred    = self.uv[:data.yd_s]/data.vtau**2
        vw_dplus_pred    = self.vw[:data.yd_s]/data.vtau**2
        uw_dplus_pred    = self.uw[:data.yd_s]/data.vtau**2
        uv_uplus_pred    = -np.flip(self.uv[data.yu_s:])/data.vtau**2
        vw_uplus_pred    = -np.flip(self.vw[data.yu_s:])/data.vtau**2
        uw_uplus_pred    = np.flip(self.uw[data.yu_s:])/data.vtau**2
        import matplotlib.pyplot as plt
        from matplotlib import cm 
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass 
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20
        plt.figure()
        plt.plot(data.yplus,uurms_dplus_data,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(data.yplus,uurms_uplus_data,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(data.yplus,uurms_dplus_pred,'-^',color=cmap[3,:],label='CNN lower')
        plt.plot(data.yplus,uurms_uplus_pred,'--^',color=cmap[3,:],label='CNN upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$u\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_u.png')
        plt.figure()
        plt.plot(data.yplus,vvrms_dplus_data,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(data.yplus,vvrms_uplus_data,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(data.yplus,vvrms_dplus_pred,'-^',color=cmap[3,:],label='CNN lower')
        plt.plot(data.yplus,vvrms_uplus_pred,'--^',color=cmap[3,:],label='CNN upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$v\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_v.png')
        plt.figure()
        plt.plot(data.yplus,wwrms_dplus_data,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(data.yplus,wwrms_uplus_data,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(data.yplus,wwrms_dplus_pred,'-^',color=cmap[3,:],label='CNN lower')
        plt.plot(data.yplus,wwrms_uplus_pred,'--^',color=cmap[3,:],label='CNN upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$w\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_w.png')
        
        plt.figure()
        plt.plot(data.yplus,uv_dplus_data,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(data.yplus,uv_uplus_data,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(data.yplus,uv_dplus_pred,'-^',color=cmap[3,:],label='CNN lower')
        plt.plot(data.yplus,uv_uplus_pred,'--^',color=cmap[3,:],label='CNN upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uv\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uv.png')
        plt.figure()
        plt.plot(data.yplus,vw_dplus_data,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(data.yplus,vw_uplus_data,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(data.yplus,vw_dplus_pred,'-^',color=cmap[3,:],label='CNN lower')
        plt.plot(data.yplus,vw_uplus_pred,'--^',color=cmap[3,:],label='CNN upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$vw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/vw.png')
        plt.figure()
        plt.plot(data.yplus,uw_dplus_data,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(data.yplus,uw_uplus_data,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(data.yplus,uw_dplus_pred,'-^',color=cmap[3,:],label='CNN lower')
        plt.plot(data.yplus,uw_uplus_pred,'--^',color=cmap[3,:],label='CNN upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uw.png')
        
    def plotrms_sim(self,data):
        """
        Function for plotting the rms
        """        
        uurms_plus_data = (data.uurms[:data.yd_s]+\
                           np.flip(data.uurms[data.yu_s:]))/2/data.vtau
        vvrms_plus_data = (data.vvrms[:data.yd_s]+\
                           np.flip(data.vvrms[data.yu_s:]))/2/data.vtau
        wwrms_plus_data = (data.wwrms[:data.yd_s]+\
                           np.flip(data.wwrms[data.yu_s:]))/2/data.vtau
        uv_plus_data    = (data.uv[:data.yd_s]-\
                           np.flip(data.uv[data.yu_s:]))/2/data.vtau**2
        vw_plus_data    = (data.vw[:data.yd_s]-\
                           np.flip(data.vw[data.yu_s:]))/2/data.vtau**2
        uw_plus_data    = (data.uw[:data.yd_s]+\
                           np.flip(data.uw[data.yu_s:]))/2/data.vtau**2
        uurms_plus_pred = (self.uurms[:data.yd_s]+
                           np.flip(self.uurms[data.yu_s:]))/2/data.vtau
        vvrms_plus_pred = (self.vvrms[:data.yd_s]+\
                           np.flip(self.vvrms[data.yu_s:]))/2/data.vtau
        wwrms_plus_pred = (self.wwrms[:data.yd_s]+\
                           np.flip(self.wwrms[data.yu_s:]))/2/data.vtau
        uv_plus_pred    = (self.uv[:data.yd_s]-\
                           np.flip(self.uv[data.yu_s:]))/2/data.vtau**2
        vw_plus_pred    = (self.vw[:data.yd_s]-\
                           np.flip(self.vw[data.yu_s:]))/2/data.vtau**2
        uw_plus_pred    = (self.uw[:data.yd_s]+\
                           np.flip(self.uw[data.yu_s:]))/2/data.vtau**2
        import matplotlib.pyplot as plt
        from matplotlib import cm 
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass 
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20
        plt.figure()
        plt.plot(data.yplus,uurms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uurms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$u\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_u.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(uurms_plus_data-uurms_plus_pred,\
                                          uurms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{u\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_u_e.png')
        plt.figure()
        plt.plot(data.yplus,vvrms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,vvrms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$v\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_v.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(vvrms_plus_data-vvrms_plus_pred,\
                                          vvrms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{v\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_v_e.png')
        plt.figure()
        plt.plot(data.yplus,wwrms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,wwrms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$w\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_w.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(wwrms_plus_data-wwrms_plus_pred,\
                                          wwrms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{w\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_w_e.png')
        
        plt.figure()
        plt.plot(data.yplus,uv_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uv_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uv\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.xscale('log')
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uv.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(uv_plus_data-uv_plus_pred,\
                                          uv_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{uv\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_uv_e.png')
        plt.figure()
        plt.plot(data.yplus,vw_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,vw_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$../../results/Simulation_3d/vw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/vw.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(vw_plus_data-vw_plus_pred,\
                                          vw_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{vw\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_vw_e.png')
        plt.figure()
        plt.plot(data.yplus,uw_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uw_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uw.png') 
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(uw_plus_data-uw_plus_pred,\
                                          uw_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{uw\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_uw_e.png')  
        
    def plotrms_simlin(self,data):
        """
        Function for plotting the rms
        """        
        uurms_plus_data = (data.uurms[:data.yd_s]+\
                           np.flip(data.uurms[data.yu_s:]))/2/data.vtau
        vvrms_plus_data = (data.vvrms[:data.yd_s]+\
                           np.flip(data.vvrms[data.yu_s:]))/2/data.vtau
        wwrms_plus_data = (data.wwrms[:data.yd_s]+\
                           np.flip(data.wwrms[data.yu_s:]))/2/data.vtau
        uv_plus_data    = (data.uv[:data.yd_s]-\
                           np.flip(data.uv[data.yu_s:]))/2/data.vtau**2
        vw_plus_data    = (data.vw[:data.yd_s]-\
                           np.flip(data.vw[data.yu_s:]))/2/data.vtau**2
        uw_plus_data    = (data.uw[:data.yd_s]+\
                           np.flip(data.uw[data.yu_s:]))/2/data.vtau**2
        uurms_plus_pred = (self.uurms[:data.yd_s]+
                           np.flip(self.uurms[data.yu_s:]))/2/data.vtau
        vvrms_plus_pred = (self.vvrms[:data.yd_s]+\
                           np.flip(self.vvrms[data.yu_s:]))/2/data.vtau
        wwrms_plus_pred = (self.wwrms[:data.yd_s]+\
                           np.flip(self.wwrms[data.yu_s:]))/2/data.vtau
        uv_plus_pred    = (self.uv[:data.yd_s]-\
                           np.flip(self.uv[data.yu_s:]))/2/data.vtau**2
        vw_plus_pred    = (self.vw[:data.yd_s]-\
                           np.flip(self.vw[data.yu_s:]))/2/data.vtau**2
        uw_plus_pred    = (self.uw[:data.yd_s]+\
                           np.flip(self.uw[data.yu_s:]))/2/data.vtau**2
        import matplotlib.pyplot as plt
        from matplotlib import cm
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass  
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20
        plt.figure()
        plt.plot(data.yplus,uurms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uurms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$u\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([0,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_ulin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((uurms_plus_data-uurms_plus_pred),\
                 uurms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{u\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([0,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_uline.png')
        plt.figure()
        plt.plot(data.yplus,vvrms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,vvrms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$v\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([0,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_vlin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((vvrms_plus_data-vvrms_plus_pred),\
                 vvrms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{v\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([0,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_vline.png')
        plt.figure()
        plt.plot(data.yplus,wwrms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,wwrms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$w\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([0,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_wlin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((wwrms_plus_data-wwrms_plus_pred),\
                 wwrms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{w\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([0,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_wline.png')
        
        plt.figure()
        plt.plot(data.yplus,uv_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uv_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uv\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([0,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uvlin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((uv_plus_data-uv_plus_pred),\
                 uv_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{uv\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([0,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uvline.png')
        plt.figure()
        plt.plot(data.yplus,vw_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,vw_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$vw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([0,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/vwlin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((vw_plus_data-vw_plus_pred),\
                 vw_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{vw\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([0,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/vwline.png')
        plt.figure()
        plt.plot(data.yplus,uw_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uw_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([0,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uwlin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((uw_plus_data-uw_plus_pred),\
                 uw_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{uw\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([0,125])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uwline.png')
        
    def plot_flowfield(self,data,ii,value,axis='y',down_y=1,down_z=1,down_x=1,\
                       facerr=1,padpix=15):
        """
        Function for saving the flowfield in an axis and value
        """
        # Read the fields
        try:
            data.UUmean
        except:
            data.read_Umean()
        try:
            data.uumin
            data.uumax
            data.vvmin
            data.vvmax
            data.wwmin
            data.wwmax
            data.uudmin
            data.uudmax
            data.vvdmin
            data.vvdmax
            data.wwdmin
            data.wwdmax
        except:
            data.read_norm()
        pfield = self.eval_model(ii,down_y=down_y,down_z=down_z,\
                                 down_x=down_x,start=ii,padpix=padpix)
        uu_p = pfield[:,:,:,0]
        vv_p = pfield[:,:,:,1]
        ww_p = pfield[:,:,:,2]
        uu_s,vv_s,ww_s = data.read_velocity(ii+1)
        self._func_flowplot(uu_p,vv_p,ww_p,uu_s,vv_s,ww_s,data,value,\
                            ii,axis=axis,facerr=facerr)
        
    def _func_flowplot(self,uu_p,vv_p,ww_p,uu_s,vv_s,ww_s,data,value,ii,\
                       axis='y',facerr=1):
        # Choose axis
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass
        if axis == 'y':
            # Calculate the nearest grid points
            near_y,index_y = nearest(data.yplus,value)
            ystr = '{0:.2f}'.format(near_y)
            # Extract the u,v,w velocities in the desired conditions
            uu_yp_d = uu_p[index_y,:,:]/data.vtau
            vv_yp_d = vv_p[index_y,:,:]/data.vtau
            ww_yp_d = ww_p[index_y,:,:]/data.vtau
            uu_ys_d = uu_s[index_y,:,:]/data.vtau
            vv_ys_d = vv_s[index_y,:,:]/data.vtau
            ww_ys_d = ww_s[index_y,:,:]/data.vtau
            uu_yp_u = uu_p[-1-index_y,:,:]/data.vtau
            vv_yp_u = vv_p[-1-index_y,:,:]/data.vtau
            ww_yp_u = ww_p[-1-index_y,:,:]/data.vtau
            uu_ys_u = uu_s[-1-index_y,:,:]/data.vtau
            vv_ys_u = vv_s[-1-index_y,:,:]/data.vtau
            ww_ys_u = ww_s[-1-index_y,:,:]/data.vtau
            uv_ys_d = np.multiply(uu_ys_d,vv_ys_d)
            vw_ys_d = np.multiply(vv_ys_d,ww_ys_d)
            uw_ys_d = np.multiply(uu_ys_d,ww_ys_d)
            uv_ys_u = np.multiply(uu_ys_u,-vv_ys_u)
            vw_ys_u = np.multiply(-vv_ys_u,ww_ys_u)
            uw_ys_u = np.multiply(uu_ys_u,ww_ys_u)
            uv_yp_d = np.multiply(uu_yp_d,vv_yp_d)
            vw_yp_d = np.multiply(vv_yp_d,ww_yp_d)
            uw_yp_d = np.multiply(uu_yp_d,ww_yp_d)
            uv_yp_u = np.multiply(uu_yp_u,-vv_yp_u)
            vw_yp_u = np.multiply(-vv_yp_u,ww_yp_u)
            uw_yp_u = np.multiply(uu_yp_u,ww_yp_u)
            uu_max = data.uumax[0]/data.vtau
            uu_min = data.uumin[0]/data.vtau
            vv_max = data.vvmax[0]/data.vtau
            vv_min = data.vvmin[0]/data.vtau
            ww_max = data.wwmax[0]/data.vtau
            ww_min = data.wwmin[0]/data.vtau
            uv_max = data.uvmax[0]/data.vtau**2
            uv_min = data.uvmin[0]/data.vtau**2
            vw_max = data.vwmax[0]/data.vtau**2
            vw_min = data.vwmin[0]/data.vtau**2
            uw_max = data.uwmax[0]/data.vtau**2
            uw_min = data.uvmin[0]/data.vtau**2
            # Calculate the errors
            error_uu_d = abs((uu_yp_d-uu_ys_d)/np.max([uu_max,uu_min]))
            error_uu_u = abs((uu_yp_u-uu_ys_u)/np.max([uu_max,uu_min]))
            error_vv_d = abs((vv_yp_d-vv_ys_d)/np.max([vv_max,vv_min]))
            error_vv_u = abs((vv_yp_u-vv_ys_u)/np.max([vv_max,vv_min]))
            error_ww_d = abs((ww_yp_d-ww_ys_d)/np.max([ww_max,ww_min]))
            error_ww_u = abs((ww_yp_u-ww_ys_u)/np.max([ww_max,ww_min]))
            error_uv_d = abs((uv_yp_d-uv_ys_d)/np.max([uv_max,uv_min]))
            error_uv_u = abs((uv_yp_u-uv_ys_u)/np.max([uv_max,uv_min]))
            error_vw_d = abs((vw_yp_d-vw_ys_d)/np.max([vw_max,vw_min]))
            error_vw_u = abs((vw_yp_u-vw_ys_u)/np.max([vw_max,vw_min]))
            error_uw_d = abs((uw_yp_d-uw_ys_d)/np.max([uw_max,uw_min]))
            error_uw_u = abs((uw_yp_u-uw_ys_u)/np.max([uw_max,uw_min]))
            # Define parameters for plots
            import matplotlib.pyplot as plt
            import matplotlib
            fs = 16  
            colormap = 'viridis'
            colormap2 = 'Greys'
            # Create the mesh
            xx,zz = np.meshgrid(data.xplus,data.zplus)
            # Plots for u
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,8))
            im0=axes[0,0].pcolor(xx,zz,uu_ys_d,vmax=uu_max,vmin=uu_min,cmap=colormap)
            axes[0,0].set_ylabel('$z^+$',fontsize=fs)
            axes[0,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,0].set_aspect('equal')
            axes[0,0].tick_params(bottom = False, labelbottom = False)
            axes[0,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[0,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[0,0].set_title('Lower channel, $y^+=$'+ystr,fontsize=fs)
            axes[0,0].text(-350,200,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[0,1].pcolor(xx,zz,uu_ys_u,vmax=uu_max,vmin=uu_min,cmap=colormap)
            axes[0,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,1].set_aspect('equal')
            axes[0,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            axes[0,1].set_title('Upper channel, $y^+=$'+ystr,fontsize=fs)
            axes[1,0].pcolor(xx,zz,uu_yp_d,vmax=uu_max,vmin=uu_min,cmap=colormap)
            axes[1,0].set_ylabel('$z^+$',fontsize=fs)
            axes[1,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,0].set_aspect('equal')
            axes[1,0].tick_params(bottom=False,labelbottom=False)
            axes[1,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[1,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[1,0].text(-350,200,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1,1].pcolor(xx,zz,uu_yp_u,vmax=uu_max,vmin=uu_min,cmap=colormap)
            axes[1,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,1].set_aspect('equal')
            axes[1,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$u^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2,0].pcolor(xx,zz,error_uu_d,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,0].set_xlabel('$x^+$',fontsize=fs)
            axes[2,0].set_ylabel('$z^+$',fontsize=fs)
            axes[2,0].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,0].set_aspect('equal')   
            axes[2,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[2,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[2,0].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,0].set_xticklabels(['$0$','$125\pi$','$250\pi$'])
            axes[2,0].text(-350,200,'Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[2,1].pcolor(xx,zz,error_uu_u,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,1].set_xlabel('$x^+$',fontsize=fs)
            axes[2,1].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,1].set_aspect('equal')   
            axes[2,1].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,1].set_xticklabels(['$0$','$125\pi$','$250\pi$'])    
            axes[2,1].tick_params(left = False, labelleft = False)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(u_p^+-u_s^+ \right)/ max(u_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
#                from os import mkdir
                os.mkdir('../../results/Simulation_3d/y+='+ystr)
            except:
                pass
            plt.savefig('../../results/Simulation_3d/y+='+ystr+'/u_'+str(ii))
            # Plots for v
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,8))
            im0=axes[0,0].pcolor(xx,zz,vv_ys_d,vmax=vv_max,vmin=vv_min,cmap=colormap)
            axes[0,0].set_ylabel('$z^+$',fontsize=fs)
            axes[0,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,0].set_aspect('equal')
            axes[0,0].tick_params(bottom = False, labelbottom = False)
            axes[0,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[0,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[0,0].set_title('Lower channel, $y^+=$'+ystr,fontsize=fs)
            axes[0,0].text(-350,200,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[0,1].pcolor(xx,zz,vv_ys_u,vmax=vv_max,vmin=vv_min,cmap=colormap)
            axes[0,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,1].set_aspect('equal')
            axes[0,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            axes[0,1].set_title('Upper channel, $y^+=$'+ystr,fontsize=fs)
            axes[1,0].pcolor(xx,zz,vv_yp_d,vmax=vv_max,vmin=vv_min,cmap=colormap)
            axes[1,0].set_ylabel('$z^+$',fontsize=fs)
            axes[1,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,0].set_aspect('equal')
            axes[1,0].tick_params(bottom=False,labelbottom=False)
            axes[1,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[1,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[1,0].text(-350,200,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1,1].pcolor(xx,zz,vv_yp_u,vmax=vv_max,vmin=vv_min,cmap=colormap)
            axes[1,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,1].set_aspect('equal')
            axes[1,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$v^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2,0].pcolor(xx,zz,error_vv_d,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,0].set_xlabel('$x^+$',fontsize=fs)
            axes[2,0].set_ylabel('$z^+$',fontsize=fs)
            axes[2,0].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,0].set_aspect('equal')   
            axes[2,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[2,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[2,0].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,0].set_xticklabels(['$0$','$125\pi$','$250\pi$'])
            axes[2,0].text(-350,200,'Rel. Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[2,1].pcolor(xx,zz,error_vv_u,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,1].set_xlabel('$x^+$',fontsize=fs)
            axes[2,1].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,1].set_aspect('equal')   
            axes[2,1].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,1].set_xticklabels(['$0$','$125\pi$','$250\pi$'])    
            axes[2,1].tick_params(left = False, labelleft = False)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(v_p^+-v_s^+ \right)/ max(v_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
#                from os import mkdir
                os.mkdir('../../results/Simulation_3d/y+='+ystr)
            except:
                pass
            plt.savefig('../../results/Simulation_3d/y+='+ystr+'/v_'+str(ii))
            # Plots for w
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,8))
            im0=axes[0,0].pcolor(xx,zz,ww_ys_d,vmax=ww_max,vmin=ww_min,cmap=colormap)
            axes[0,0].set_ylabel('$z^+$',fontsize=fs)
            axes[0,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,0].set_aspect('equal')
            axes[0,0].tick_params(bottom = False, labelbottom = False)
            axes[0,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[0,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[0,0].set_title('Lower channel, $y^+=$'+ystr,fontsize=fs)
            axes[0,0].text(-350,200,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[0,1].pcolor(xx,zz,ww_ys_u,vmax=ww_max,vmin=ww_min,cmap=colormap)
            axes[0,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,1].set_aspect('equal')
            axes[0,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            axes[0,1].set_title('Upper channel, $y^+=$'+ystr,fontsize=fs)
            axes[1,0].pcolor(xx,zz,ww_yp_d,vmax=ww_max,vmin=ww_min,cmap=colormap)
            axes[1,0].set_ylabel('$z^+$',fontsize=fs)
            axes[1,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,0].set_aspect('equal')
            axes[1,0].tick_params(bottom=False,labelbottom=False)
            axes[1,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[1,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[1,0].text(-350,200,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1,1].pcolor(xx,zz,ww_yp_u,vmax=ww_max,vmin=ww_min,cmap=colormap)
            axes[1,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,1].set_aspect('equal')
            axes[1,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$w^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2,0].pcolor(xx,zz,error_ww_d,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,0].set_xlabel('$x^+$',fontsize=fs)
            axes[2,0].set_ylabel('$z^+$',fontsize=fs)
            axes[2,0].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,0].set_aspect('equal')   
            axes[2,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[2,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[2,0].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,0].set_xticklabels(['$0$','$125\pi$','$250\pi$'])
            axes[2,0].text(-350,200,'Rel. Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[2,1].pcolor(xx,zz,error_ww_u,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,1].set_xlabel('$x^+$',fontsize=fs)
            axes[2,1].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,1].set_aspect('equal')   
            axes[2,1].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,1].set_xticklabels(['$0$','$125\pi$','$250\pi$'])    
            axes[2,1].tick_params(left = False, labelleft = False)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(w_p^+-w_s^+ \right)/ max(w_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
#                from os import mkdir
                os.mkdir('../../results/Simulation_3d/y+='+ystr)
            except:
                pass
            plt.savefig('../../results/Simulation_3d/y+='+ystr+'/w_'+str(ii))
            # Plots for uv
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,8))
            im0=axes[0,0].pcolor(xx,zz,uv_ys_d,vmax=uv_max,vmin=uv_min,cmap=colormap)
            axes[0,0].set_ylabel('$z^+$',fontsize=fs)
            axes[0,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,0].set_aspect('equal')
            axes[0,0].tick_params(bottom = False, labelbottom = False)
            axes[0,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[0,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[0,0].set_title('Lower channel, $y^+=$'+ystr,fontsize=fs)
            axes[0,0].text(-350,200,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[0,1].pcolor(xx,zz,uv_ys_u,vmax=uv_max,vmin=uv_min,cmap=colormap)
            axes[0,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,1].set_aspect('equal')
            axes[0,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            axes[0,1].set_title('Upper channel, $y^+=$'+ystr,fontsize=fs)
            axes[1,0].pcolor(xx,zz,uv_yp_d,vmax=uv_max,vmin=uv_min,cmap=colormap)
            axes[1,0].set_ylabel('$z^+$',fontsize=fs)
            axes[1,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,0].set_aspect('equal')
            axes[1,0].tick_params(bottom=False,labelbottom=False)
            axes[1,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[1,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[1,0].text(-350,200,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1,1].pcolor(xx,zz,uv_yp_u,vmax=uv_max,vmin=uv_min,cmap=colormap)
            axes[1,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,1].set_aspect('equal')
            axes[1,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$uv^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2,0].pcolor(xx,zz,error_uv_d,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,0].set_xlabel('$x^+$',fontsize=fs)
            axes[2,0].set_ylabel('$z^+$',fontsize=fs)
            axes[2,0].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,0].set_aspect('equal')   
            axes[2,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[2,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[2,0].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,0].set_xticklabels(['$0$','$125\pi$','$250\pi$'])
            axes[2,0].text(-350,200,'Rel. Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[2,1].pcolor(xx,zz,error_uv_u,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,1].set_xlabel('$x^+$',fontsize=fs)
            axes[2,1].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,1].set_aspect('equal')   
            axes[2,1].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,1].set_xticklabels(['$0$','$125\pi$','$250\pi$'])    
            axes[2,1].tick_params(left = False, labelleft = False)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(uv_p^+-uv_s^+ \right)/ max(uv_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
#                from os import mkdir
                os.mkdir('../../results/Simulation_3d/y+='+ystr)
            except:
                pass
            plt.savefig('../../results/Simulation_3d/y+='+ystr+'/uv_'+str(ii))
            # Plots for vw
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,8))
            im0=axes[0,0].pcolor(xx,zz,vw_ys_d,vmax=vw_max,vmin=vw_min,cmap=colormap)
            axes[0,0].set_ylabel('$z^+$',fontsize=fs)
            axes[0,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,0].set_aspect('equal')
            axes[0,0].tick_params(bottom = False, labelbottom = False)
            axes[0,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[0,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[0,0].set_title('Lower channel, $y^+=$'+ystr,fontsize=fs)
            axes[0,0].text(-350,200,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[0,1].pcolor(xx,zz,vw_ys_u,vmax=vw_max,vmin=vw_min,cmap=colormap)
            axes[0,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,1].set_aspect('equal')
            axes[0,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            axes[0,1].set_title('Upper channel, $y^+=$'+ystr,fontsize=fs)
            axes[1,0].pcolor(xx,zz,vw_yp_d,vmax=vw_max,vmin=vw_min,cmap=colormap)
            axes[1,0].set_ylabel('$z^+$',fontsize=fs)
            axes[1,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,0].set_aspect('equal')
            axes[1,0].tick_params(bottom=False,labelbottom=False)
            axes[1,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[1,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[1,0].text(-350,200,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1,1].pcolor(xx,zz,vw_yp_u,vmax=vw_max,vmin=vw_min,cmap=colormap)
            axes[1,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,1].set_aspect('equal')
            axes[1,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$vw^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2,0].pcolor(xx,zz,error_vw_d,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,0].set_xlabel('$x^+$',fontsize=fs)
            axes[2,0].set_ylabel('$z^+$',fontsize=fs)
            axes[2,0].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,0].set_aspect('equal')   
            axes[2,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[2,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[2,0].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,0].set_xticklabels(['$0$','$125\pi$','$250\pi$'])
            axes[2,0].text(-350,200,'Rel. Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[2,1].pcolor(xx,zz,error_vw_u,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,1].set_xlabel('$x^+$',fontsize=fs)
            axes[2,1].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,1].set_aspect('equal')   
            axes[2,1].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,1].set_xticklabels(['$0$','$125\pi$','$250\pi$'])    
            axes[2,1].tick_params(left = False, labelleft = False)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(vw_p^+-vw_s^+ \right)/ max(vw_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
#                from os import mkdir
                os.mkdir('../../results/Simulation_3d/y+='+ystr)
            except:
                pass
            plt.savefig('../../results/Simulation_3d/y+='+ystr+'/vw_'+str(ii))
            # Plots for uw
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,8))
            im0=axes[0,0].pcolor(xx,zz,uw_ys_d,vmax=uw_max,vmin=uw_min,cmap=colormap)
            axes[0,0].set_ylabel('$z^+$',fontsize=fs)
            axes[0,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,0].set_aspect('equal')
            axes[0,0].tick_params(bottom = False, labelbottom = False)
            axes[0,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[0,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[0,0].set_title('Lower channel, $y^+=$'+ystr,fontsize=fs)
            axes[0,0].text(-350,200,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[0,1].pcolor(xx,zz,uw_ys_u,vmax=uw_max,vmin=uw_min,cmap=colormap)
            axes[0,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[0,1].set_aspect('equal')
            axes[0,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            axes[0,1].set_title('Upper channel, $y^+=$'+ystr,fontsize=fs)
            axes[1,0].pcolor(xx,zz,uw_yp_d,vmax=uw_max,vmin=uw_min,cmap=colormap)
            axes[1,0].set_ylabel('$z^+$',fontsize=fs)
            axes[1,0].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,0].set_aspect('equal')
            axes[1,0].tick_params(bottom=False,labelbottom=False)
            axes[1,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[1,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[1,0].text(-350,200,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1,1].pcolor(xx,zz,uw_yp_u,vmax=uw_max,vmin=uw_min,cmap=colormap)
            axes[1,1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1,1].set_aspect('equal')
            axes[1,1].tick_params(left = False, labelleft = False, \
                bottom = False, labelbottom = False)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$uw^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2,0].pcolor(xx,zz,error_uw_d,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,0].set_xlabel('$x^+$',fontsize=fs)
            axes[2,0].set_ylabel('$z^+$',fontsize=fs)
            axes[2,0].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,0].set_aspect('equal')   
            axes[2,0].set_yticks([0,np.pi*data.rey/2,np.pi*data.rey])
            axes[2,0].set_yticklabels(['$0$','$62.5\pi$','$125\pi$'])
            axes[2,0].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,0].set_xticklabels(['$0$','$125\pi$','$250\pi$'])
            axes[2,0].text(-350,200,'Rel. Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[2,1].pcolor(xx,zz,error_uw_u,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2,1].set_xlabel('$x^+$',fontsize=fs)
            axes[2,1].tick_params(axis='both', which='major', labelsize=fs)
            axes[2,1].set_aspect('equal')   
            axes[2,1].set_xticks([0,np.pi*data.rey,2*np.pi*data.rey])
            axes[2,1].set_xticklabels(['$0$','$125\pi$','$250\pi$'])    
            axes[2,1].tick_params(left = False, labelleft = False)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(uw_p^+-uw_s^+ \right)/ max(uw^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
#                from os import mkdir
                os.mkdir('../../results/Simulation_3d/y+='+ystr)
            except:
                pass
            plt.savefig('../../results/Simulation_3d/y+='+ystr+'/uw_'+str(ii))
        
        
    def mre_pred(self,data,start,end,step=1,down_y=1,down_z=1,down_x=1,\
                 padpix=15,delta_t=1):
        """
        Function for calculating the mean relative error
        """
        from time import sleep
        try:
            data.UUmean
        except:
            data.read_Umean()
        try:
            data.uumin
            data.uumax
            data.vvmin
            data.vvmax
            data.wwmin
            data.wwmax
            data.uudmin
            data.uudmax
            data.vvdmin
            data.vvdmax
            data.wwdmin
            data.wwdmax
        except:
            data.read_norm()
        for ii in range(start,end,step):
            pfield = self.eval_model(ii,down_y=down_y,down_z=down_z,\
                                     down_x=down_x,start=ii,padpix=padpix)
            uu_p = pfield[:,:,:,0]
            vv_p = pfield[:,:,:,1]
            ww_p = pfield[:,:,:,2]
            uu_s,vv_s,ww_s = data.read_velocity(ii+delta_t)
            error_uu = abs(uu_p-uu_s)/np.max([abs(data.uumax),abs(data.uumin)])
            error_vv = abs(vv_p-vv_s)/np.max([abs(data.vvmax),abs(data.vvmin)])
            error_ww = abs(ww_p-ww_s)/np.max([abs(data.wwmax),abs(data.wwmin)])
            if ii==start:
                error_uu_cum = np.sum(np.multiply(error_uu,data.vol))
                error_vv_cum = np.sum(np.multiply(error_vv,data.vol))
                error_ww_cum = np.sum(np.multiply(error_ww,data.vol))
                vol_cum = np.sum(data.vol)
            else:
                error_uu_cum += np.sum(np.multiply(error_uu,data.vol))
                error_vv_cum += np.sum(np.multiply(error_vv,data.vol))
                error_ww_cum += np.sum(np.multiply(error_ww,data.vol))
                vol_cum += np.sum(data.vol)
            print('err_u: '+str(np.sum(np.multiply(error_uu,data.vol))/np.sum(data.vol))+\
                  'err_v: '+str(np.sum(np.multiply(error_vv,data.vol))/np.sum(data.vol))+\
                  'err_w: '+str(np.sum(np.multiply(error_ww,data.vol))/np.sum(data.vol)))
            sleep(0.5)
        self.mre_uu = error_uu_cum/vol_cum
        self.mre_vv = error_vv_cum/vol_cum
        self.mre_ww = error_ww_cum/vol_cum
        print("Error u: " + str(self.mre_uu))
        print("Error v: " + str(self.mre_vv))
        print("Error w: " + str(self.mre_ww))
        

    
            
    def savemre(self,file="../../results/Simulation_3d/mre_predic.txt"):
        """
        Function for saving the value of the rms velocity
        """
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass
        file_save = open(file, "w+")           
        content = "Error u: " + str(self.mre_uu) + '\n'
        file_save.write(content)    
        content = "Error v: " + str(self.mre_vv) + '\n'
        file_save.write(content)    
        content = "Error w: " + str(self.mre_ww) + '\n'
        file_save.write(content)         

