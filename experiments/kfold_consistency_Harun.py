# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '../utils/')
from harun import *
sys.path.insert(1, '../utils/')
from metrics import *
from pathlib import Path
import cv2
import imutils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


#STRUCTURE: TE

path=Path('../dataset/')
path_tot=path/'train'
path_test=path/'test'
path_gt=path/'GT_TE'
path_models=Path('../models/dataset/TE')


x,y=load_imgs(path_tot,path_gt,'TE')
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
model= build_unet(input_shape=(256,256,1))
callbacks = [
    EarlyStopping(patience=15),
    ReduceLROnPlateau(factor=0.05, patience=5)]

fold_no = 1
metricas=[]
for train, test in kfold.split(x):
    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]
    
    x_train, mean, std = std_norm(x_train)
    x_test = std_norm_test(x_test, mean, std)
    
    x_train, y_train = data_augmentation(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=13)
    
    x_train = x_train.astype("float32") 
    y_train = y_train.astype("float32")/255  

    x_test = x_test.astype("float32") 
    y_test = y_test.astype("float32")/255  
    
    train_gen = DataGenerator(x_train, y_train, 16)
    model= build_unet(input_shape=(256,256,1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=my_loss_fn, metrics=[jaccard_index])
    
    print(f'Training for fold {fold_no} ...')
    history = model.fit(train_gen,batch_size=16, epochs=200, callbacks=callbacks, validation_data=(x_test, y_test))
    predictions = model.predict(x_test)
    model.save('fold'+str(fold_no))    
    fold_no = fold_no + 1
    metricas.append(summary_metrics(y_test,predictions, 0.5))

accuracy = [el['accuracy'] for el in metricas]
precision = [el['precision'] for el in metricas]
recall = [el['recall'] for el in metricas]
specificity = [el['specificity'] for el in metricas]
jaccard = [el['jaccard'] for el in metricas]
dice = [el['dice'] for el in metricas]

print('TE structure')
print('accuracy: ',np.mean(accuracy),' ', np.std(accuracy))
print('precision: ', np.mean(precision), ' ',np.std(precision))
print('recall: ',np.mean(recall),' ', np.std(recall))
print('specificity: ',np.mean(specificity),' ',  np.std(specificity))
print('jaccard: ',np.mean(jaccard),' ', np.std(jaccard))
print('dice: ',np.mean(dice),' ',  np.std(dice))

#STRUCTURE: ICM

path=Path('../dataset/')
path_tot=path/'IMAGES'
path_gt=path/'GT_ICM'
path_models=Path('../models/dataset/ICM')


x,y=load_imgs(path_tot,path_gt,'ICM')
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
model= build_unet(input_shape=(256,256,1))
callbacks = [
    EarlyStopping(patience=15),
    ReduceLROnPlateau(factor=0.05, patience=5)]

fold_no = 1
metricas=[]
for train, test in kfold.split(x):
    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]
    
    x_train, mean, std = std_norm(x_train)
    x_test = std_norm_test(x_test, mean, std)
    
    x_train, y_train = data_augmentation(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=13)
    
    x_train = x_train.astype("float32") 
    y_train = y_train.astype("float32")/255  

    x_test = x_test.astype("float32") 
    y_test = y_test.astype("float32")/255  
    
    train_gen = DataGenerator(x_train, y_train, 16)
    model= build_unet(input_shape=(256,256,1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=my_loss_fn, metrics=[jaccard_index])
    
    print(f'Training for fold {fold_no} ...')
    history = model.fit(train_gen,batch_size=16, epochs=200, callbacks=callbacks, validation_data=(x_test, y_test))
    predictions = model.predict(x_test)
    model.save('fold'+str(fold_no))    
    fold_no = fold_no + 1
    metricas.append(summary_metrics(y_test,predictions, 0.5))

accuracy = [el['accuracy'] for el in metricas]
precision = [el['precision'] for el in metricas]
recall = [el['recall'] for el in metricas]
specificity = [el['specificity'] for el in metricas]
jaccard = [el['jaccard'] for el in metricas]
dice = [el['dice'] for el in metricas]

print('ICM structure')
print('accuracy: ',np.mean(accuracy),' ', np.std(accuracy))
print('precision: ', np.mean(precision), ' ',np.std(precision))
print('recall: ',np.mean(recall),' ', np.std(recall))
print('specificity: ',np.mean(specificity),' ',  np.std(specificity))
print('jaccard: ',np.mean(jaccard),' ', np.std(jaccard))
print('dice: ',np.mean(dice),' ',  np.std(dice))
