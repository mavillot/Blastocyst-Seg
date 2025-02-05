# -*- coding: utf-8 -*-
import torch
import cv2
import sys
import numpy as np
import pandas as pd

from utils.crop import crop_embyo
from utils.preds import prediction


if len(sys.argv) > 2:
    path_video = sys.argv[1]
    path_model = sys.argv[2]
else:
    print("No values provided.")


# CAMBIAR PARA INTRODUCIR 
#path_video=Path("video.mp4")
#path_model='models/models/hrnet.pth'

# Load the trained model
model = torch.jit.load(path_model)
model = model.cpu()


def areas(video_path,tam=500,N=10):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    xmin,xmax,ymin,ymax=crop_embyo(frame,tam=500)
    frame=frame[ymin:ymax, xmin:xmax]
    H, W, _ = frame.shape
    TE,ZP,ICM=[],[],[]
    while ret:
        mask=prediction(frame, model)
        ZP.append(sum(sum(mask==75)))
        TE.append(sum(sum(mask==255)))
        ICM.append(sum(sum(mask==150)))
        ret, frame = cap.read()
        if ret: frame=frame[ymin:ymax, xmin:xmax]
    return ZP,TE,ICM


def intersection(te, icm):
    # return the exact point of the intersection between TE and ICM after the max is obtained
    N=40
    te=np.convolve(te, np.ones(N)/N, mode='same')
    icm=np.convolve(icm, np.ones(N)/N, mode='same')
    intersections=[]
    maxi=list(icm).index(max(icm))
    for i in range (1,len(te)):
        if te[i-1]<icm[i-1] and te[i]>icm[i] and i>maxi:
            intersections.append(i)
    if intersections!=[]:
        pos=intersections[0]
    else:
        pos=len(te) - 2
    return pos     

def real_time_blasto(TE,ICM,fps):
    # return the real time when the blastocyst formation take place in hours
    # 1 video sec = 3000 real secs
    k=intersection(TE, ICM)
    return k*5/(fps*6)

def video_time_blasto(TE,ICM,fps):
    # return the video time when the blastocyst formation take place in minutes.seconds
    k=intersection(TE, ICM)
    seconds=k/fps
    segs=seconds%60
    mins=int(seconds/60)
    return mins+segs/100

cap = cv2.VideoCapture(path_video)
fps=cap.get(cv2.CAP_PROP_FPS)

ZP,TE,ICM=areas(path_video)
real=real_time_blasto(TE,ICM,fps)
video=video_time_blasto(TE,ICM,fps)

#SAVE THE INFORMATION
data_areas=pd.DataFrame({'ZP':ZP,'TE':TE,'ICM':ICM})
data_areas.to_csv('areas.csv', index=False)
file=open('blastocyst_formation.txt','w')
texto='VIDEO TIME (minutes.secs): '+ str(video) + '\nREAL TIME (hours): '+ str(real)
file.write(texto)

