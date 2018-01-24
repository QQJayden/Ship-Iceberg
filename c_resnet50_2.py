# https://www.kaggle.com/devm2024/transfer-learning-with-vgg-16-cnn-aug-lb-0-1712
#Mandatory imports
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

#Import Keras.
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))


def img_resize(images,img_size=197):
    imgs=[]
    for img in images:
        new_img=cv2.resize(img,(img_size,img_size))
        imgs.append(new_img)
    imgs=np.array(imgs)
    return imgs

train = pd.read_json("./Data/train.json")
test = pd.read_json("./Data/test.json")
target_train=train['is_iceberg']

train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')#We have only 133 NAs.
# 缺失入射角填0
train['inc_angle']=train['inc_angle'].fillna(0)
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')

X_angle=train['inc_angle']
X_test_angle=test['inc_angle']

#Generate the training data
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_band_3=(X_band_1-X_band_2)
#X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis]
                          , X_band_2[:, :, :, np.newaxis]
                         , X_band_3[:, :, :, np.newaxis]], axis=-1)


X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3=(X_band_test_1-X_band_test_2)
#X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , X_band_test_3[:, :, :, np.newaxis]], axis=-1)

#resize
X_train=img_resize(X_train)
X_test =img_resize(X_test)

batch_size = 32
# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip=False,
                         vertical_flip=False,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         channel_shift_range=0,
                         zoom_range=0.2,
                         rotation_range=20)#20


# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    g_seed=93
    genX1 = gen.flow(X1, y, batch_size=batch_size, seed=g_seed)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=g_seed)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        # Assert arrays are equal - this was for peace of mind, but slows down training
        # np.testing.assert_array_equal(X1i[0],X2i[0])
        yield [X1i[0], X2i[1]], X1i[1]


# Finally create generator
def get_callbacks(filepath, pa=10):
    es = EarlyStopping('val_loss', patience=pa, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


def getVggAngleModel():
    input_2 = Input(shape=(1,), name="angle")
    angle_layer = Dense(1, )(input_2)
    input_1 = Input(shape=X_train.shape[1:], name='img')
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
    x = base_model(input_1)
    x = Flatten()(x)

    merge_one = concatenate([x, angle_layer])
    merge_one = Dropout(0.1)(merge_one)
    predictions = Dense(1, activation='sigmoid')(merge_one)
    model = Model(inputs=[input_1, input_2], outputs=predictions)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model


# Using K-fold Cross Validation with Data Augmentation.
def myAngleCV(X_train, X_angle, X_test):
    kfold = 4
    folds = list(StratifiedKFold(n_splits=kfold, shuffle=True, random_state=2017).split(X_train, target_train))

    y_test_pred_log = 0
    y_valid_pred_log = 0.0 * target_train
    train_loss=[]
    valid_loss=[]

    for j, (train_idx, test_idx) in enumerate(folds):
        print('\nFold {}/{} :'.format(j+1,kfold))
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]

        # Angle
        X_angle_cv = X_angle[train_idx]
        X_angle_hold = X_angle[test_idx]

        # define file path and get callbacks
        file_path = "./model/{}fold_resnet50_model_weights.hdf5".format(j+1)
        callbacks = get_callbacks(filepath=file_path)
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
        galaxyModel = getVggAngleModel()
        galaxyModel.fit_generator(
            gen_flow,
            steps_per_epoch=int(X_train_cv.shape[0]/batch_size),
            epochs=100,
            shuffle=True,
            verbose=2,
            validation_data=([X_holdout, X_angle_hold], Y_holdout),
            callbacks=callbacks)

        # Getting the Best Model
        galaxyModel.load_weights(filepath=file_path)
        # Getting Training Score
        score = galaxyModel.evaluate([X_train_cv, X_angle_cv], y_train_cv, verbose=0)
        train_loss.append(score[0])
        print('     Fold{} Train loss:{}, Train accuracy:{}'.format(j+1,score[0],score[1]))
        # Getting Valid Score
        score = galaxyModel.evaluate([X_holdout, X_angle_hold], Y_holdout, verbose=0)
        valid_loss.append(score[0])
        print('     Fold{} Valid loss:{}, Valid accuracy:{}'.format(j+1,score[0], score[1]))

        # Getting OOF Prediction
        pred_valid = galaxyModel.predict([X_holdout, X_angle_hold])
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        # Getting Test Prediction
        temp_test = galaxyModel.predict([X_test, X_test_angle])
        y_test_pred_log += temp_test.reshape(temp_test.shape[0])

    # mean loss for each fold
    train_loss,valid_loss=np.array(train_loss),np.array(valid_loss)
    mean_train_loss,mean_valid_loss=round(train_loss.mean(),6),round(valid_loss.mean(),6)
    std_train_loss,std_valid_loss=round(train_loss.std(),6),round(valid_loss.std(),6)
    overfit_coe = (mean_valid_loss - mean_train_loss) / mean_train_loss
    print()
    print('mean train loss:{} ± {}  mean valid loss:{} ± {}'.format(mean_train_loss,std_train_loss,mean_valid_loss,std_valid_loss))

    y_test_pred_log = y_test_pred_log / kfold
    print('Whole Train Data Log Loss: {}, overfit_coe: {} '.format(log_loss(target_train, y_valid_pred_log),overfit_coe))
    return y_valid_pred_log,y_test_pred_log

train_preds,test_preds=myAngleCV(X_train, X_angle, X_test)

#full prediction
full_predcit=pd.Series(np.r_[train_preds,test_preds])

# #Submission
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=test_preds
# full_predcit.to_csv('./stacks/c_resnet50_2input_3.csv', index=False)
submission.to_csv('./sub/c_resnet50_2input_3.csv', index=False)



#保持原来的数据增强,dropout0.1






#dropout 0.2
#mean train loss:0.107145 ± 0.05952  mean valid loss:0.240455 ± 0.03151
#Whole Train Data Log Loss: 0.24040779823823266, overfit_coe: 1.2442017826310139


#augument2 scale[0.8,1.1]
#mean train loss:0.093316 ± 0.028896  mean valid loss:0.240397 ± 0.035437
#Whole Train Data Log Loss: 0.2403952663159272, overfit_coe: 1.5761605726777832



#resnet50_1.csv LB
#mean train loss:0.05657 ± 0.027089  mean valid loss:0.220485 ± 0.012639
#Whole Train Data Log Loss: 0.22047018831126042, overfit_coe: 2.8975605444581927

#resnet50_2.csv LB     flip,flop
#mean train loss:0.164402 ± 0.021002  mean valid loss:0.223812 ± 0.013438
#Whole Train Data Log Loss: 0.22383318331329538, overfit_coe: 0.3613702996313915

#只做水平翻转
#an train loss:0.112439 ± 0.038756  mean valid loss:0.248148 ± 0.019189
#Whole Train Data Log Loss: 0.248150343924097, overfit_coe: 1.206956660945046



