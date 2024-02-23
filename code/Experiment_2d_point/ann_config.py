# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:12:37 2023

@author: andres cremades botella

file containing the functions to configurate the CNN
"""
import numpy as np

def nearest(array,value):
    array_value = abs(array-value)
    index = np.argmin(array_value)
    nearest = array[index]
    return nearest,index

def plottrain(file="../../results/Experiment_2d/hist.txt"):
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
    plt.savefig('../../results/Experiment_2d/Loss_plot.png')
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
    from tensorflow.keras.layers import Conv2D, BatchNormalization,\
        Activation
    xx = Conv2D(nfil, kernel_size=kernel, 
                strides=(stride,stride),padding="same")(xx)
    xx = BatchNormalization()(xx) 
    xx = Activation(activ)(xx)
    return xx

def blockpool(xx,nfil,stride,activ,kernel):
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
    from tensorflow.keras.layers import Conv2D, BatchNormalization,\
        Activation,AveragePooling2D
    xx = Conv2D(nfil, kernel_size=kernel, 
                strides=(stride,stride),padding="same")(xx)
    xx = BatchNormalization()(xx) 
    xx = Activation(activ)(xx)
    xx = AveragePooling2D(2,strides=1,padding='same')(xx)
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
    from tensorflow.keras.layers import Conv2D, BatchNormalization,\
        Activation
    xx = Conv2D(nfil, kernel_size=kernel, 
                strides=(stride,stride),padding="same")(xx)
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
    from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization,\
        Activation 
    xx = Conv2DTranspose(nfil, kernel_size=kernel,
                         strides=(stride,stride),padding="valid",output_padding=outpad)(xx)
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
    from tensorflow.keras.layers import Conv2D, BatchNormalization,\
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
    def __init__(self,ngpu=1,fileddbb='../../data/uv_fields_io/PIV',\
                 pond='none'):
        self.devices(ngpu)
        self.fileddbb = fileddbb
        self.pond = pond
        try:
            os.mkdir('../../results/Experiment_2d/')
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
        from tensorflow.keras.layers import Input,MaxPool2D,\
        Concatenate,Add,Activation
        from tensorflow.image import crop_to_bounding_box
        import tensorflow as tf
        dim0 = shp[0]
        dim1 = shp[1]
        dim2 = shp[2]
        shp = (dim0,dim1,dim2)
        self.inputs = Input(shape=shp)

        # First layer
        xx11 = block(self.inputs,nfil[0],stride[0],activ[0],kernel[0]) 
        xx12 = block(xx11,nfil[0],stride[0],activ[0],kernel[0])  
        xx13 = block(xx12,nfil[0],stride[0],activ[0],kernel[0])   
        # to second layer
        xx20 = MaxPool2D(2)(xx13)
        # second layer
        xx21 = block(xx20,nfil[1],stride[1],activ[1],kernel[1])
        xx22 = block(xx21,nfil[1],stride[1],activ[1],kernel[1])
        xx23 = block(xx22,nfil[1],stride[1],activ[1],kernel[1])
        # to third layer
        xx30 = MaxPool2D(2)(xx23)
        # third layer
        xx31 = block(xx30,nfil[2],stride[2],activ[2],kernel[2])
        xx32 = block(xx31,nfil[2],stride[2],activ[2],kernel[2])
        xx33 = block(xx32,nfil[2],stride[2],activ[2],kernel[2])
        # to fourth layer
        xx40 = MaxPool2D(2)(xx33)
        # fourth layer
        xx41 = block(xx40,nfil[2]*2,stride[2],activ[2],kernel[2])
        xx42 = block(xx41,nfil[2]*2,stride[2],activ[2],kernel[2])
        xx43 = block(xx42,nfil[2]*2,stride[2],activ[2],kernel[2])
        # go to third layer
        xx30b = invblock(xx43,nfil[2],2,activ[2],kernel[2],outpad=(0,0))
        xx31b = Concatenate()([xx30b[:,:-1,:,:],xx33])
        # third         
        xx32b = block(xx31b,nfil[1],stride[1],activ[1],kernel[1])
        xx33b = block(xx32b,nfil[1],stride[1],activ[1],kernel[1])
        # go to second layer
        xx20b = invblock(xx33b,nfil[1],2,activ[1],kernel[2],outpad=(0,0))
        xx21b = Concatenate()([xx20b,xx23])   
        # second         
        xx22b = block(xx21b,nfil[1],stride[1],activ[1],kernel[1])
        xx23b = block(xx22b,nfil[1],stride[1],activ[1],kernel[1])
        # go to first layer
        xx10b = invblock(xx23b,nfil[0],2,activ[0],kernel[1],outpad=(0,0)) 
        xx11b = Concatenate()([xx10b,xx13])
        # First layer
        xx12b = block(xx11b,nfil[0],stride[0],activ[0],kernel[0])
        xx13b = block(xx12b,nfil[0],stride[0],activ[0],kernel[0])
        xx00b = block(xx13b,2,stride[0],activ[0],kernel[0])
        #
        xx01b = xx00b[:,padpix:-padpix,:,:]
        self.outputs = xx01b



    
    def define_model(self,shp=(2416,215,2),nfil=np.array([32,64,96]),\
                     stride=np.array([1,1,1]),\
                     activ=["relu","relu","relu"],\
                     kernel=[(3,3),(3,3),(3,3)],optmom=0.9,\
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
        print('Using Keras version: ', tf.keras.__version__)
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
                    batch_size=1,down_y=1,down_x=1,\
                    trainfile='../../results/Experiment_2d/trained_model.h5',trainhist='../../results/Experiment_2d/hist.txt',\
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
        data.geom_param(start,down_y,down_x)
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
                     
#%%            
    def load_ANN(self,filename='../../results/Experiment_2d/trained_model.h5'):
        """ 
        Function for loading the ANN model
        """
        import tensorflow as tf 
        self.model = tf.keras.models.load_model(filename)
    
    def load_model(self,filename='../../results/Experiment_2d/trained_model.h5'): 
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
        print('Using Keras version: ', tf.keras.__version__)
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
        uu,vv = data.read_velocity(index,padpix=padpix)
        past_field = data.norm_velocity(uu,vv)
        pred_field = self.model.predict(past_field)
        pred_field_dim = data.dimensional_velocity(pred_field)
        past_field_dim = np.zeros((data.mx,data.my,2))
        past_field_dim[:,:,0] = uu
        past_field_dim[:,:,1] = vv
        self.my = data.my
        self.mx = data.mx
        return pred_field_dim
        
    def _calc_rms(self,uu,vv,uu2_cum=0,vv2_cum=0,uv_cum=0,nn_cum=0):
        uu2 = np.multiply(uu,uu)
        vv2 = np.multiply(vv,vv)
        uv  = np.multiply(uu,vv)
        uu2_cum += np.sum(uu2,axis=(0))
        vv2_cum += np.sum(vv2,axis=(0))
        uv_cum  += np.sum(uv,axis=(0))
        nn_cum += np.ones((self.my,))*self.mx
        return uu2_cum,vv2_cum,uv_cum,nn_cum
    
    def _calc_rms_xy(self,uu,vv,uu2_cum=0,vv2_cum=0,uv_cum=0,nn_cum=0):
        uu2 = np.multiply(uu,uu)
        vv2 = np.multiply(vv,vv)
        uv  = np.multiply(uu,vv)
        uu2_cum += uu2
        vv2_cum += vv2
        uv_cum  += uv
        nn_cum += np.ones((len(uu[:,0]),len(uu[0,:])))
        return uu2_cum,vv2_cum,uv_cum,nn_cum
    
    
    def pred_rms(self,start,end,step=1,down_y=1,down_x=1,padpix=15):
        """
        Function for calculating the rms of the velocity components and the 
        product of the velocity fluctuations of the predicted fields
        """
        for ii in range(start,end,step):
            pfield = self.eval_model(ii,down_y=down_y,\
                                     down_x=down_x,start=start,padpix=padpix)
            uu  = pfield[:,:,0]
            vv  = pfield[:,:,1]
            if ii==start:
                uu2_cum,vv2_cum,uv_cum,nn_cum =\
                self._calc_rms(uu,vv)
            else:
                uu2_cum,vv2_cum,uv_cum,nn_cum =\
                self._calc_rms(uu,vv,uu2_cum=uu2_cum,vv2_cum=vv2_cum,\
                               uv_cum=uv_cum,nn_cum=nn_cum)
        self.uurms = np.sqrt(np.divide(uu2_cum,nn_cum))    
        self.vvrms = np.sqrt(np.divide(vv2_cum,nn_cum)) 
        self.uv    = np.divide(uv_cum,nn_cum)
   
    
    def pred_rms_xy(self,start=1,end=2,step=1,down_y=1,down_x=1,padpix=15,\
                    testcases=False,filetest='../../results/Experiment_2d/ind_val.txt'):
        """
        Function for calculating the rms of the velocity components and the 
        product of the velocity fluctuations of the predicted fields
        """
        import get_data_fun as gd
        data = gd.get_data_norm(self.fileddbb,pond=self.pond)
        data.geom_param(start,down_y,down_x)
        try:
            data.read_norm()
        except:
            data.calc_norm(start,end)
        if testcases:
            file_read = open(filetest,"r")
            listcases = np.array(file_read.readline().replace('[','').\
                                 replace(']','').split(','),dtype='int')[::step]
        else:
            listcases = range(start,end,step)
        for ii in listcases:
            try:
                uu_s,vv_s = data.read_velocity(ii,out=True,padpix=padpix)
                pfield = self.eval_model(ii,down_y=down_y,\
                                         down_x=down_x,start=start,padpix=padpix)
                flag = 1
            except:
                flag = 0
            if flag == 1:
                uu  = pfield[:,:,0]
                vv  = pfield[:,:,1]
                if ii==listcases[0]:
                    uu2_cum,vv2_cum,uv_cum,nn_cum =\
                    self._calc_rms_xy(uu,vv)
                    uu2s_cum,vv2s_cum,uvs_cum,nns_cum =\
                    self._calc_rms_xy(uu_s,vv_s)
                else:
                    uu2_cum,vv2_cum,uv_cum,nn_cum =\
                    self._calc_rms_xy(uu,vv,uu2_cum=uu2_cum,vv2_cum=vv2_cum,\
                                   uv_cum=uv_cum,nn_cum=nn_cum)
                    uu2s_cum,vv2s_cum,uvs_cum,nns_cum =\
                    self._calc_rms_xy(uu_s,vv_s,uu2_cum=uu2s_cum,vv2_cum=vv2s_cum,\
                                   uv_cum=uvs_cum,nn_cum=nns_cum)
        self.uurms_xy = np.sqrt(np.divide(uu2_cum,nn_cum))    
        self.vvrms_xy = np.sqrt(np.divide(vv2_cum,nn_cum)) 
        self.uv_xy    = np.divide(uv_cum,nn_cum)   
        self.uurms_xys = np.sqrt(np.divide(uu2s_cum,nns_cum))    
        self.vvrms_xys = np.sqrt(np.divide(vv2s_cum,nns_cum)) 
        self.uv_xys    = np.divide(uvs_cum,nns_cum)         
        
            
    def saverms(self,file="../../results/Experiment_2d/Urmspred.txt"):
        """
        Function for saving the value of the rms velocity
        """
        file_save = open(file, "w+")           
        content = str(self.uurms.tolist())+'\n'
        file_save.write(content)    
        content = str(self.vvrms.tolist())+'\n'
        file_save.write(content)          
        content = str(self.uv.tolist())+'\n'
        file_save.write(content)  
        
    def saverms_xy(self,file="../../results/Experiment_2d/Urmspred_xy.h5"):
        """
        Function for saving the value of the rms velocity
        """
        import h5py
        hf = h5py.File(file, 'w')
        hf.create_dataset('uurms', data=self.uurms_xy)
        hf.create_dataset('vvrms', data=self.vvrms_xy)
        hf.create_dataset('uv', data=self.uv_xy)
        hf.create_dataset('uurms_s', data=self.uurms_xys)
        hf.create_dataset('vvrms_s', data=self.vvrms_xys)
        hf.create_dataset('uv_s', data=self.uv_xys)
        
    def readrms(self,file="../../results/Experiment_2d/Urmspred.txt"):
        """
        Read the predicted rms velocity
        """
        file_read = open(file,"r")
        self.uurms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvrms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uv = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
    
    def readrms_xy(self,file="../../results/Experiment_2d/Urmspred_xy.h5"):
        """
        Read the predicted rms velocity
        """
        import h5py
        hf = h5py.File(file, 'r')
        self.uurms_xy =  np.array(hf['uurms'])
        self.vvrms_xy =  np.array(hf['vvrms'])
        self.uv_xy =  np.array(hf['uv'])
        self.uurms_xys =  np.array(hf['uurms_s'])
        self.vvrms_xys =  np.array(hf['vvrms_s'])
        self.uv_xys =  np.array(hf['uv_s'])
        
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
        plt.savefig('../../results/Experiment_2d/rms_u.png')
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
        plt.savefig('../../results/Experiment_2d/rms_v.png')
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
        plt.savefig('../../results/Experiment_2d/rms_w.png')
        
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
        plt.savefig('../../results/Experiment_2d/uv.png')
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
        plt.savefig('../../results/Experiment_2d/vw.png')
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
        plt.savefig('../../results/Experiment_2d/uw.png')
 
        
    def plotrms_sim_xy(self,data,padpix=15,colormap='viridis'):
        """
        Function for plotting the rms
        """        
        yy,xx = np.meshgrid(data.yplus,data.xplus[padpix:-padpix])
        uurmsmax= np.max(np.abs(self.uurms_xys))
        vvrmsmax= np.max(np.abs(self.vvrms_xys))
        uvrmsmax= np.max(np.abs(self.uv_xys))
        if padpix > 0:
            vol_pad = data.vol[padpix:-padpix,:]
        else:
            vol_pad = data.vol
        err_uurms_plus_pred = np.abs(self.uurms_xy-self.uurms_xys)/uurmsmax
        err_vvrms_plus_pred = np.abs(self.vvrms_xy-self.vvrms_xys)/vvrmsmax
        err_uv_plus_pred    = np.abs(self.uv_xy-self.uv_xys)/uvrmsmax
        err_uurms_vol = np.sum(np.multiply(err_uurms_plus_pred,vol_pad))
        err_vvrms_vol = np.sum(np.multiply(err_vvrms_plus_pred,vol_pad))
        err_uv_vol = np.sum(np.multiply(err_uv_plus_pred,vol_pad))
        voltot = np.sum(vol_pad)
        err_uurms_vol /= voltot
        err_vvrms_vol /= voltot
        err_uv_vol /= voltot
        print('Error in u\''': '+str(err_uurms_vol))
        print('Error in v\''': '+str(err_vvrms_vol))
        print('Error in uv\''': '+str(err_uv_vol))
        import matplotlib.pyplot as plt
        from matplotlib import cm  
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20
        fig=plt.figure()
        im0 = plt.pcolor(xx,yy,err_uurms_plus_pred)
        plt.xlabel('$x^+$',fontsize=fs)
        plt.ylabel('$y^+$',fontsize=fs)
        cb = fig.colorbar(im0,orientation="vertical",aspect=20)
        cb.outline.set_visible(False)
        cb.set_label(label=r"$((u\')_p^+-(u\')_s^+)/max((u\')_s^+)$",fontsize=fs)
        cb.ax.tick_params(axis="both",labelsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_u_xy.png')
        fig=plt.figure()
        im0 = plt.pcolor(xx,yy,err_vvrms_plus_pred)
        plt.xlabel('$x^+$',fontsize=fs)
        plt.ylabel('$y^+$',fontsize=fs)
        cb = fig.colorbar(im0,orientation="vertical",aspect=20)
        cb.outline.set_visible(False)
        cb.set_label(label=r"$((v\')_p^+-(v\')_s^+)/max((v\')_s^+)$",fontsize=fs)
        cb.ax.tick_params(axis="both",labelsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_v_xy.png')
        fig=plt.figure()
        im0 = plt.pcolor(xx,yy,err_uv_plus_pred)
        plt.xlabel('$x^+$',fontsize=fs)
        plt.ylabel('$y^+$',fontsize=fs)
        cb = fig.colorbar(im0,orientation="vertical",aspect=20)
        cb.outline.set_visible(False)
        cb.set_label(label=r"$((uv\')_p^+-(uv\')_s^+)/max((uv\')_s^+)$",fontsize=fs)
        cb.ax.tick_params(axis="both",labelsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/uv_xy.png')
        l_x = data.mx-2*padpix
        err_x1_u = err_uurms_plus_pred[int(l_x/6)]
        err_x2_u = err_uurms_plus_pred[int(l_x/6*2)]
        err_x3_u = err_uurms_plus_pred[int(l_x/6*3)]
        err_x4_u = err_uurms_plus_pred[int(l_x/6*4)]
        err_x5_u = err_uurms_plus_pred[int(l_x/6*5)] 
        err_x1_v = err_vvrms_plus_pred[int(l_x/6)]
        err_x2_v = err_vvrms_plus_pred[int(l_x/6*2)]
        err_x3_v = err_vvrms_plus_pred[int(l_x/6*3)]
        err_x4_v = err_vvrms_plus_pred[int(l_x/6*4)]
        err_x5_v = err_vvrms_plus_pred[int(l_x/6*5)]      
        x1 = int(data.xplus[int(data.mx/6)])
        x2 = int(data.xplus[int(data.mx/6*2)])
        x3 = int(data.xplus[int(data.mx/6*3)])
        x4 = int(data.xplus[int(data.mx/6*4)])
        x5 = int(data.xplus[int(data.mx/6*5)])
        fig=plt.figure()
        plt.plot(data.yplus,err_x1_u,label='$x^+=$'+str(x1),color=plt.cm.get_cmap(colormap,6).colors[0,:])
        plt.plot(data.yplus,err_x2_u,label='$x^+=$'+str(x2),color=plt.cm.get_cmap(colormap,6).colors[1,:])
        plt.plot(data.yplus,err_x3_u,label='$x^+=$'+str(x3),color=plt.cm.get_cmap(colormap,6).colors[2,:])
        plt.plot(data.yplus,err_x4_u,label='$x^+=$'+str(x4),color=plt.cm.get_cmap(colormap,6).colors[3,:])
        plt.plot(data.yplus,err_x5_u,label='$x^+=$'+str(x5),color=plt.cm.get_cmap(colormap,6).colors[4,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel(r"$((u\')_p^+-(u\')_s^+)/max((u\')_s^+)$",fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.grid()
        plt.savefig('../../results/Experiment_2d/error_u_posx.png')
        fig=plt.figure()
        plt.plot(data.yplus,err_x1_v,label='$x^+=$'+str(x1),color=plt.cm.get_cmap(colormap,6).colors[0,:])
        plt.plot(data.yplus,err_x2_v,label='$x^+=$'+str(x2),color=plt.cm.get_cmap(colormap,6).colors[1,:])
        plt.plot(data.yplus,err_x3_v,label='$x^+=$'+str(x3),color=plt.cm.get_cmap(colormap,6).colors[2,:])
        plt.plot(data.yplus,err_x4_v,label='$x^+=$'+str(x4),color=plt.cm.get_cmap(colormap,6).colors[3,:])
        plt.plot(data.yplus,err_x5_v,label='$x^+=$'+str(x5),color=plt.cm.get_cmap(colormap,6).colors[4,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel(r"$((v\')_p^+-(v\')_s^+)/max((v\')_s^+)$",fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.grid()
        plt.savefig('../../results/Experiment_2d/error_v_posx.png')
        
               
    def plotrms_sim_xy_compare(self,data,xplus=0,colormap='viridis',padpix=15,\
                               literature=False,lit_file='../../results/Experiment_2d/torroja_retau_934.txt'):
        """
        Function for plotting the rms
        """        
        index_x = np.argmin(abs(data.xplus-xplus))
        yy,xx = np.meshgrid(data.yplus,data.xplus)
        uurms_pred = self.uurms_xy[index_x-padpix,:]
        vvrms_pred = self.vvrms_xy[index_x-padpix,:]
        uv_pred = self.uv_xy[index_x-padpix,:]
        uvrms_pred = np.multiply(uurms_pred,vvrms_pred)
        uurms_sim = self.uurms_xys[index_x-padpix,:]
        vvrms_sim = self.vvrms_xys[index_x-padpix,:]
        uv_sim = self.uv_xys[index_x-padpix,:]
        uvrms_sim = np.multiply(uurms_sim,vvrms_sim)
        uurms_tot = data.uurms_point[index_x,:]
        vvrms_tot = data.vvrms_point[index_x,:]
        uv_tot = data.uv_point[index_x,:]
        uvrms_tot = np.multiply(uurms_tot,vvrms_tot)
        if literature:
            colormax = 5
            yplus_lite = []
            uplus_lite = []
            vplus_lite = []
            with open(lit_file) as ff:
                lines = ff.readlines()
                for line in lines:
                    if line[0] == '%':
                        continue
                    else:
                        line_split = line.split()
                        yplus_lite.append(float(line_split[1]))
                        uplus_lite.append(float(line_split[3]))
                        vplus_lite.append(float(line_split[4]))
            yplus_lite = np.array(yplus_lite)
            uplus_lite = np.array(uplus_lite)
            vplus_lite = np.array(vplus_lite)
            uvplus_lite = np.multiply(uplus_lite,vplus_lite)            
        else:
            colormax = 4
        import matplotlib.pyplot as plt
        from matplotlib import cm  
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20    
        xxlab = data.xplus[index_x]
        fig=plt.figure()
        plt.plot(data.yplus,uurms_pred/data.vtau,label='$prediction$',color=plt.cm.get_cmap(colormap,colormax).colors[0,:])
        plt.plot(data.yplus,uurms_sim/data.vtau,label='$simulation$',color=plt.cm.get_cmap(colormap,colormax).colors[1,:])
        plt.plot(data.yplus,uurms_tot/data.vtau,label='$simulation complete$',color=plt.cm.get_cmap(colormap,colormax).colors[2,:])
        if literature:
            plt.plot(yplus_lite,uplus_lite,label='$literature$',color=plt.cm.get_cmap(colormap,colormax).colors[3,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel(r"$(u\')^+$",fontsize=fs)
        plt.title(r"$x^+ = $"+str(np.round(xxlab)))
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.grid()
        plt.savefig('../../results/Experiment_2d/urms_x'+str(np.round(xxlab))+'.png')
        fig=plt.figure()
        plt.plot(data.yplus,vvrms_pred/data.vtau,label='$prediction$',color=plt.cm.get_cmap(colormap,colormax).colors[0,:])
        plt.plot(data.yplus,vvrms_sim/data.vtau,label='$simulation$',color=plt.cm.get_cmap(colormap,colormax).colors[1,:])
        plt.plot(data.yplus,vvrms_tot/data.vtau,label='$simulation complete$',color=plt.cm.get_cmap(colormap,colormax).colors[2,:])
        if literature:
            plt.plot(yplus_lite,vplus_lite,label='$literature$',color=plt.cm.get_cmap(colormap,colormax).colors[3,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel(r"$(v\')^+$",fontsize=fs)
        plt.title(r"$x^+ = $"+str(np.round(xxlab)))
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.grid()
        plt.savefig('../../results/Experiment_2d/vrms_x'+str(np.round(xxlab))+'.png')
        fig=plt.figure()
        plt.plot(data.yplus,abs(uv_pred/data.vtau**2),label='$prediction$',color=plt.cm.get_cmap(colormap,colormax).colors[0,:])
        plt.plot(data.yplus,abs(uv_sim/data.vtau**2),label='$simulation$',color=plt.cm.get_cmap(colormap,colormax).colors[1,:])
        plt.plot(data.yplus,abs(uv_tot/data.vtau**2),label='$simulation complete$',color=plt.cm.get_cmap(colormap,colormax).colors[2,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel(r"$(uv)^+$",fontsize=fs)
        plt.title(r"$x^+ = $"+str(np.round(xxlab)))
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.grid()
        plt.savefig('../../results/Experiment_2d/uv_x'+str(np.round(xxlab))+'.png')
        fig=plt.figure()
        plt.plot(data.yplus,abs(uvrms_pred/data.vtau**2),label='$prediction$',color=plt.cm.get_cmap(colormap,colormax).colors[0,:])
        plt.plot(data.yplus,abs(uvrms_sim/data.vtau**2),label='$simulation$',color=plt.cm.get_cmap(colormap,colormax).colors[1,:])
        plt.plot(data.yplus,abs(uvrms_tot/data.vtau**2),label='$simulation complete$',color=plt.cm.get_cmap(colormap,colormax).colors[2,:])
        if literature:
            plt.plot(yplus_lite,uvplus_lite,label='$literature$',color=plt.cm.get_cmap(colormap,colormax).colors[3,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel(r"$(u'v')^+$",fontsize=fs)
        plt.title(r"$x^+ = $"+str(np.round(xxlab)))
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.grid()
        plt.savefig('../../results/Experiment_2d/urmsvrms_x'+str(np.round(xxlab))+'.png')
        
    def plotrms_sim(self,data):
        """
        Function for plotting the rms
        """        
        uurms_plus_data = data.uurms/data.vtau
        vvrms_plus_data = data.vvrms/data.vtau
        uv_plus_data    = data.uv/data.vtau**2
        uurms_plus_pred = self.uurms/data.vtau
        vvrms_plus_pred = self.vvrms/data.vtau
        uv_plus_pred    = self.uv/data.vtau**2
        import matplotlib.pyplot as plt
        from matplotlib import cm  
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
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_u.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(uurms_plus_data-uurms_plus_pred,\
                                          uurms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{u\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_u_e.png')
        plt.figure()
        plt.plot(data.yplus,vvrms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,vvrms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$v\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_v.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(vvrms_plus_data-vvrms_plus_pred,\
                                          vvrms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{v\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_v_e.png')
        plt.figure()
        plt.plot(data.yplus,uv_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uv_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uv\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.xscale('log')
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/uv.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide(uv_plus_data-uv_plus_pred,\
                                          uv_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{uv\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_uv_e.png')
        
    def plotrms_simlin(self,data):
        """
        Function for plotting the rms
        """        
        uurms_plus_data = data.uurms/data.vtau
        vvrms_plus_data = data.vvrms/data.vtau
        uv_plus_data    = data.uv/data.vtau**2
        uurms_plus_pred = self.uurms/data.vtau
        vvrms_plus_pred = self.vvrms/data.vtau
        uv_plus_pred    = self.uv/data.vtau**2
        import matplotlib.pyplot as plt
        from matplotlib import cm  
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
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_ulin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((uurms_plus_data-uurms_plus_pred),\
                 uurms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{u\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([300,7500])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_uline.png')
        plt.figure()
        plt.plot(data.yplus,vvrms_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,vvrms_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$v\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_vlin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((vvrms_plus_data-vvrms_plus_pred),\
                 vvrms_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{v\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([300,7500])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_vline.png')
        plt.figure()
        plt.plot(data.yplus,uv_plus_data,'-',color=cmap[0,:],label='DNS')
        plt.plot(data.yplus,uv_plus_pred,'-',color=cmap[3,:],label='CNN')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uv\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/uvlin.png')
        plt.figure()
        plt.plot(data.yplus,abs(np.divide((uv_plus_data-uv_plus_pred),\
                 uv_plus_data)),'-',color=cmap[0,:])
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\epsilon_{uv\'^+}$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xlim([300,7500])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/uvline.png')
        
    def plot_flowfield(self,data,ii,axis='y',down_y=1,down_x=1,\
                       facerr=1,padpix=15):
        """
        Function for saving the flowfield in an axis and value
        """
        # Read the fields
        try:
            data.uumin
            data.uumax
            data.vvmin
            data.vvmax
        except:
            data.read_norm()
        pfield = self.eval_model(ii,down_y=down_y,\
                                 down_x=down_x,start=ii,padpix=padpix)
        uu_p = pfield[:,:,0]
        vv_p = pfield[:,:,1]
        uu_s,vv_s = data.read_velocity(ii+1,padpix=padpix,out=True)
        self._func_flowplot(uu_p,vv_p,uu_s,vv_s,data,\
                            ii,axis=axis,facerr=facerr,padpix=padpix)
        
    def _func_flowplot(self,uu_p,vv_p,uu_s,vv_s,data,ii,\
                       axis='y',facerr=1,padpix=15):
        # Choose axis
        if axis == 'y':
            # Calculate the nearest grid points
            # Extract the u,v,w velocities in the desired conditions
            uu_yp = uu_p/data.vtau
            vv_yp = vv_p/data.vtau
            uu_ys = uu_s/data.vtau
            vv_ys = vv_s/data.vtau
            uv_ys = np.multiply(uu_ys,vv_ys)
            uv_yp = np.multiply(uu_yp,vv_yp)
            uu_max = data.uumax[0]/data.vtau
            uu_min = data.uumin[0]/data.vtau
            vv_max = data.vvmax[0]/data.vtau
            vv_min = data.vvmin[0]/data.vtau
            uv_max = data.uvmax[0]/data.vtau**2
            uv_min = data.uvmin[0]/data.vtau**2
            # Calculate the errors
            error_uu = abs((uu_yp-uu_ys)/np.max([uu_max,uu_min]))
            error_vv = abs((vv_yp-vv_ys)/np.max([vv_max,vv_min]))
            error_uv = abs((uv_yp-uv_ys)/np.max([uv_max,uv_min]))
            # Define parameters for plots
            import matplotlib.pyplot as plt
            import matplotlib
            fs = 16  
            colormap = 'viridis'
            colormap2 = 'Greys'
            # Create the mesh
            yy,xx = np.meshgrid(data.yplus,data.xplus[padpix:-padpix])
            # Plots for u
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,5))
            im0=axes[0].pcolor(xx,yy,uu_ys,vmax=uu_max,vmin=uu_min,cmap=colormap)
            axes[0].set_ylabel('$y^+$',fontsize=fs)
            axes[0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0].set_aspect('equal')
            axes[0].tick_params(bottom = False, labelbottom = False)
            axes[0].set_yticks([0,data.rey/2,data.rey])
            axes[0].text(-3000,500,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1].pcolor(xx,yy,uu_yp,vmax=uu_max,vmin=uu_min,cmap=colormap)
            axes[1].set_ylabel('$y^+$',fontsize=fs)
            axes[1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1].set_aspect('equal')
            axes[1].tick_params(bottom=False,labelbottom=False)
            axes[1].set_yticks([0,data.rey/2,data.rey])
            axes[1].text(-3000,500,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$u^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2].pcolor(xx,yy,error_uu,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2].set_xlabel('$x^+$',fontsize=fs)
            axes[2].set_ylabel('$y^+$',fontsize=fs)
            axes[2].tick_params(axis='both', which='major', labelsize=fs)
            axes[2].set_aspect('equal')   
            axes[2].set_yticks([0,data.rey/2,data.rey])
            axes[2].text(-3000,500,'Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(u_p^+-u_s^+ \right)/ max(u_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
                from os import mkdir
                mkdir('../../results/Experiment_2d/field_error')
            except:
                pass
            plt.savefig('../../results/Experiment_2d/field_error/u_'+str(ii))
            # Plots for v
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,5))
            im0=axes[0].pcolor(xx,yy,vv_ys,vmax=vv_max,vmin=vv_min,cmap=colormap)
            axes[0].set_ylabel('$y^+$',fontsize=fs)
            axes[0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0].set_aspect('equal')
            axes[0].tick_params(bottom = False, labelbottom = False)
            axes[0].set_yticks([0,data.rey/2,data.rey])
            axes[0].text(-3000,500,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1].pcolor(xx,yy,vv_yp,vmax=vv_max,vmin=vv_min,cmap=colormap)
            axes[1].set_ylabel('$y^+$',fontsize=fs)
            axes[1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1].set_aspect('equal')
            axes[1].tick_params(bottom=False,labelbottom=False)
            axes[1].set_yticks([0,data.rey/2,data.rey])
            axes[1].text(-3000,500,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$v^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2].pcolor(xx,yy,error_vv,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2].set_xlabel('$x^+$',fontsize=fs)
            axes[2].set_ylabel('$y^+$',fontsize=fs)
            axes[2].tick_params(axis='both', which='major', labelsize=fs)
            axes[2].set_aspect('equal')   
            axes[2].set_yticks([0,data.rey/2,data.rey])
            axes[2].text(-3000,500,'Rel. Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(v_p^+-v_s^+ \right)/ max(v_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)   
            try:
                from os import mkdir
                mkdir('../../results/Experiment_2d/field_error')
            except:
                pass
            plt.savefig('../../results/Experiment_2d/field_error/v_'+str(ii))
            # Plots for uv
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,5))
            im0=axes[0].pcolor(xx,yy,uv_ys,vmax=uv_max,vmin=uv_min,cmap=colormap)
            axes[0].set_ylabel('$y^+$',fontsize=fs)
            axes[0].tick_params(axis='both',which='major',labelsize=fs)
            axes[0].set_aspect('equal')
            axes[0].tick_params(bottom = False, labelbottom = False)
            axes[0].set_yticks([0,data.rey/2,data.rey])
            axes[0].text(-3000,500,'DNS',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            axes[1].pcolor(xx,yy,uv_yp,vmax=uv_max,vmin=uv_min,cmap=colormap)
            axes[1].set_ylabel('$y^+$',fontsize=fs)
            axes[1].tick_params(axis='both',which='major',labelsize=fs)
            axes[1].set_aspect('equal')
            axes[1].tick_params(bottom=False,labelbottom=False)
            axes[1].set_yticks([0,data.rey/2,data.rey])
            axes[1].text(-3000,500,'U-net',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            cb = fig.colorbar(im0, ax=axes.ravel().tolist(),\
                              orientation="vertical",aspect=20)
            cb.outline.set_visible(False)
            cb.set_label(label=r"$uv^+$",\
                         fontsize=fs)
            cb.ax.tick_params(axis="both",labelsize=fs)
            im2=axes[2].pcolor(xx,yy,error_uv,vmin=0,vmax=facerr,cmap=colormap2)
            axes[2].set_xlabel('$x^+$',fontsize=fs)
            axes[2].set_ylabel('$y^+$',fontsize=fs)
            axes[2].tick_params(axis='both', which='major', labelsize=fs)
            axes[2].set_aspect('equal')   
            axes[2].set_yticks([0,data.rey/2,data.rey])
            axes[2].text(-3000,500,'Rel. Error',verticalalignment='center',rotation=90,\
                fontsize=fs+2)
            cb1 = fig.colorbar(im2, ax=axes.ravel().tolist(),\
                               orientation="vertical",aspect=20)
            cb1.outline.set_visible(False)
            cb1.set_label(label=r'$ \left(uv_p^+-uv_s^+ \right)/ max(uv_s^+)$',fontsize=fs)
            cb1.ax.tick_params(axis="both",labelsize=fs)     
            try:
                from os import mkdir
                mkdir('../../results/Experiment_2d/field_error')
            except:
                pass
            plt.savefig('../../results/Experiment_2d/field_error/uv_'+str(ii))
        
        
    def mre_pred(self,data,start=1,end=2,step=1,down_y=1,down_z=1,down_x=1,\
                 padpix=15,testcases=False,filetest='../../results/Experiment_2d/ind_val.txt'):
        """
        Function for calculating the mean relative error
        """
        from time import sleep
        try:
            data.uumin
            data.uumax
            data.vvmin
            data.vvmax
        except:
            data.read_norm()
        if testcases:
            file_read = open(filetest,"r")
            listcases = np.array(file_read.readline().replace('[','').\
                                 replace(']','').split(','),dtype='int')[::step]
        else:
            listcases = range(start,end,step)
        for ii in listcases:
            pfield = self.eval_model(ii,down_y=down_y,down_z=down_z,\
                                     down_x=down_x,start=ii,padpix=padpix)
            uu_p = pfield[:,:,0]
            vv_p = pfield[:,:,1]
            uu_s,vv_s = data.read_velocity(ii+1,padpix=padpix,out=True)
            error_uu = abs(uu_p-uu_s)/np.max([abs(data.uumax),abs(data.uumin)])
            error_vv = abs(vv_p-vv_s)/np.max([abs(data.vvmax),abs(data.vvmin)])
            if padpix > 0:
                vol_pad = data.vol[padpix:-padpix,:]
            else:
                vol_pad = data.vol
            if ii==listcases[0]:
                error_uu_cum = np.sum(np.multiply(error_uu,vol_pad))
                error_vv_cum = np.sum(np.multiply(error_vv,vol_pad))
                vol_cum = np.sum(vol_pad)
            else:
                error_uu_cum += np.sum(np.multiply(error_uu,vol_pad))
                error_vv_cum += np.sum(np.multiply(error_vv,vol_pad))
                vol_cum += np.sum(vol_pad)
            print('err_u: '+str(error_uu_cum/vol_cum)+\
                  'err_v: '+str(error_vv_cum/vol_cum))
            sleep(0.5)
        self.mre_uu = error_uu_cum/vol_cum
        self.mre_vv = error_vv_cum/vol_cum
        print("Error u: " + str(self.mre_uu))
        print("Error v: " + str(self.mre_vv))
        

    
            
    def savemre(self,file="../../results/Experiment_2d/mre_predic.txt"):
        """
        Function for saving the value of the rms velocity
        """
        file_save = open(file, "w+")           
        content = "Error u: " + str(self.mre_uu) + '\n'
        file_save.write(content)    
        content = "Error v: " + str(self.mre_vv) + '\n'
        file_save.write(content)         

