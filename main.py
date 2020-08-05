# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:56:59 2020

@author: REZA
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, GRU, LSTM, RNN

#from keras.models import Sequential, model_from_json
#from keras.layers import Conv1D,Conv2D,MaxPooling1D,Flatten,Dense,Dropout,BatchNormalization, GRU, LSTM, RNN
from tensorflow.keras import regularizers as reg

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs,pick_types,events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws,read_raw_edf
from mne.datasets import eegbci


from sklearn.model_selection import train_test_split


#%%
def cnn(conv_layers=3,conv_sizes=(64,128,256),filter_size=3, fc_layers=2,fc_sizes=(4096,2048),
        dropout=0.5,pool_size=2,init='he_uniform',act='relu',optim='adam',pool=True,
        reg = reg.l2(0.05)):

    classifier = Sequential()
    for i in range(conv_layers):
        classifier.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              activation = act,kernel_initializer=init,kernel_regularizer=reg))
        classifier.add(BatchNormalization())
        if pool:
            classifier.add(MaxPooling1D(pool_size = 2))
    classifier.add(Flatten())
    for j in range(fc_layers):
        classifier.add(Dense(fc_sizes[j], activation = act,kernel_initializer=init,kernel_regularizer=reg))
        classifier.add(Dropout(dropout))
    classifier.add(Dense(4, activation = 'softmax',kernel_initializer=init))
    classifier.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=10,batch_size=64)   
    
    
#%%
def cnn_plot(conv_layers=3,conv_sizes=(64,128,256),filter_size=3, fc_layers=2,fc_sizes=(4096,2048),
        dropout=0.5,pool_size=2,init='he_uniform',act='relu',optim='adam',pool=True,
        reg = reg.l2(0.05),epochs=10):

    classifier = Sequential()
    for i in range(conv_layers):
        classifier.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              activation = act,kernel_initializer=init,kernel_regularizer=reg))
        classifier.add(BatchNormalization())
        if pool:
            classifier.add(MaxPooling1D(pool_size = 2))
    classifier.add(Flatten())
    for j in range(fc_layers):
        classifier.add(Dense(fc_sizes[j], activation = act,kernel_initializer=init,kernel_regularizer=reg))
        classifier.add(Dropout(dropout))
    classifier.add(Dense(4, activation = 'softmax',kernel_initializer=init))
    classifier.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    history = classifier.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=epochs,batch_size=64)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

#%%
subject=1
runs=[6,10,14]
raw_fnames=eegbci.load_data(subject,runs)
raw=concatenate_raws([read_raw_edf(f,preload=True) for f in raw_fnames])

#%%
eegbci.standardize(raw)
#create 10-10 system
montage=make_standard_montage('standard_1005')
raw.set_montage(montage)

# %%
raw.filter(6,13)
events,_=events_from_annotations(raw,event_id=dict(T1=0,T2=1))
#0: left, 1:right
picks=pick_types(raw.info,meg=False,eeg=True,stim=False,eog=False,exclude='bads')

# %%
tmin,tmax=-1,4
epochs=Epochs(raw,events,None,tmin,tmax,proj=True,picks=picks,
             baseline=None,preload=True)
#epochs_train=epochs.copy().crop(0,2)
epochs_train=epochs.copy().crop(0,2)
labels=epochs.events[:,-1]

# %%
epochs_data=epochs.get_data()
epochs_data_train=epochs_train.get_data()

#%%

X_train,X_test,y_train,y_test = train_test_split(
        epochs_data_train, labels, test_size=0.2, random_state=42)



#%%
cnn(conv_layers=2,conv_sizes=(64,64),fc_layers=3,fc_sizes=(1024,512,256))
#%%
cnn_plot(conv_layers=3,conv_sizes=(32,32,32),fc_layers=2,fc_sizes=(512,256),epochs=30)



