#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:18:16 2018

@author: user
"""

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.metrics import log_loss

import cv2
from sklearn.model_selection import KFold

from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D,Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Load data
train = pd.read_json("../iceberg/data/train.json")
target_train=train['is_iceberg']
# test = pd.read_json("../input/test.json")


import json
with open('./data/test_nohuman.json', 'r') as f:
    test = json.load(f)
    test=pd.DataFrame(test)
    
    
# 缺省值填充方法？？
train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')
train['inc_angle']=train['inc_angle'].fillna(method='pad')
# train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
# train['inc_angle']=train['inc_angle'].fillna(train['inc_angle'].mean())
X_angle=train['inc_angle']

test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
# test.inc_angle = test.inc_angle.astype(float).fillna(0.0)
# test['inc_angle']=test['inc_angle'].fillna(test['inc_angle'].mean())
X_test_angle=test['inc_angle']

y_train=train['is_iceberg']



def color_composite(data):
#    import cv2
#     w,h = 197,197
    w,h = 139,139
    rgb_arrays = []
#     !!! 
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
#         band_3 = band_1 / band_2
        band_3 = np.zeros((75,75))
        
#        band_3 = band_1-band_2
        
        subt = abs(band_1-band_2)
        W1 = subt/subt.max()
        W2=1-W1
        band_3=W1 * band_1 + W2 * band_2

        '''
        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))
        '''

#         rgb = np.dstack((r, g, b))
        rgb = np.dstack((band_1, band_2, band_3))
        #Add in to resize for resnet50 use 197 x 197
        rgb = cv2.resize(rgb, (w,h)).astype(np.float32)
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)

X_train = color_composite(train)
X_test = color_composite(test)


from keras.layers.merge import Concatenate
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate

#Create the model
#model = simple_cnn()
input_tensor = Input(shape=(139,139,3))

input_2 = Input(shape=[1], name="angle")
angle_layer = Dense(1, )(input_2)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(139,139,3))
x = base_model.output

#bn = BatchNormalization()(input_tensor)
#x=base_model(bn)

#x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
  

x = concatenate([x, angle_layer])
# x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
#x = Dropout(0.2)(x)
#x = Dropout(0.25)(x)
#x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
#x = Dropout(0.25)(x)
predictions = Dense(1, activation='sigmoid')(x)

#model = Model(inputs=[input_tensor,input_2], outputs=predictions)
model = Model(inputs=[base_model.input,input_2], outputs=predictions)
#for layer in base_model.layers:
#    layer.trainable = False
for layer in model.layers[:15]:
    layer.trainable = False
for layer in model.layers[15:]:
    layer.trainable = True
    

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adamax
from keras.optimizers import Nadam

# 使用不同的优化
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
adagrad = Adagrad(lr = 1e-3, epsilon = 1e-6)
rmsprop = RMSprop(lr=1e-3, rho = 0.9, epsilon=1e-6)
adadelta = Adadelta(lr=1e-3, rho=0.95, epsilon=1e-06)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator
batch_size = 32 # 原来是3
#Lets define the image transormations that we want
gen = ImageDataGenerator(horizontal_flip=False,
                         vertical_flip=False,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         rotation_range=20)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

#Finally create out generator
# gen_flow = gen_flow_for_one_inputs(X_train, y_train)

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Finally create generator
def get_callbacks(filepath, patience=2):
   es = EarlyStopping('val_loss', patience=10, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]

'''
epochs_to_wait_for_improve = 10
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
checkpoint_callback = ModelCheckpoint('./model/BestKerasModelResNet50.h5', monitor='val_loss', 
                                      verbose=1, save_best_only=True, mode='min')
'''

#Using K-fold Cross Validation with Data Augmentation.
def myAngleCV(X_train, X_angle, X_test):
# def myAngleCV(X_train, X_test):
    # K-折交叉验证
    K=4
    
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=2017).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log=0
    y_valid_pred_log = 0.0*target_train
    
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout= target_train[test_idx]
        
        #Angle
        X_angle_cv=X_angle[train_idx]
        X_angle_hold=X_angle[test_idx]

        #define file path and get callbacks
        file_path = "./model/%s_aug_InceptionV3_model_weights.hdf5"%j
        callbacks = get_callbacks(filepath=file_path, patience=5)
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
#         gen_flow = gen_flow_for_one_inputs(X_train_cv,  y_train_cv)
#         galaxyModel= getVggAngleModel()
        galaxyModel= model
    
        # 调整训练参数
        galaxyModel.fit_generator(
                gen_flow,
                steps_per_epoch=len(X_train_cv)//batch_size,
#                steps_per_epoch=24,
                #steps_per_epoch=100,
                epochs=100,
                shuffle=True,
                verbose=1,
                validation_data=([X_holdout,X_angle_hold], Y_holdout),
#                 validation_data=(X_holdout, Y_holdout),
                callbacks=callbacks)

        #Getting the Best Model
        galaxyModel.load_weights(filepath=file_path)
        #Getting Training Score
        score = galaxyModel.evaluate([X_train_cv,X_angle_cv], y_train_cv, verbose=0)
#         score = galaxyModel.evaluate(X_train_cv, y_train_cv, verbose=0)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        #Getting Test Score
        score = galaxyModel.evaluate([X_holdout,X_angle_hold], Y_holdout, verbose=0)
#         score = galaxyModel.evaluate(X_holdout, Y_holdout, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        #Getting validation Score.
        pred_valid=galaxyModel.predict([X_holdout,X_angle_hold])
#         pred_valid=galaxyModel.predict(X_holdout)
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        #Getting Test Scores
        temp_test=galaxyModel.predict([X_test, X_test_angle])
#         temp_test=galaxyModel.predict(X_test)
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

        #Getting Train Scores
        temp_train=galaxyModel.predict([X_train, X_angle])
#         temp_train=galaxyModel.predict(X_train)
        y_train_pred_log+=temp_train.reshape(temp_train.shape[0])

    y_test_pred_log=y_test_pred_log/K
    y_train_pred_log=y_train_pred_log/K
    
#    from sklearn.metrics import log_loss

    print('\n Train Log Loss Validation= ',log_loss(target_train, y_train_pred_log))
    print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
    return y_valid_pred_log,y_test_pred_log



from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


train_preds,test_preds=myAngleCV(X_train, X_angle, X_test)
# preds=myAngleCV(X_train,X_test)


#Submission for each day.
full_predict=pd.Series(np.r_[train_preds,test_preds])
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=test_preds
full_predict.to_csv('./submission/full_preds_inceptionv3.csv',index=False)
#submission.to_csv('./submission/subTLInceptionV3-6.6.csv', index=False)