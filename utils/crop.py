# -*- coding: utf-8 -*-
import numpy as np
import cv2

# RECORTAR EMBRION
def centroide(img):
    x_media,y_media,n_media=0,0,0
    Y,X=img.shape
    for i in range(X):
        for j in range(Y):
            if img[j,i]==0:
                x_media+=i
                y_media+=j
                n_media+=1
    return (int(x_media/(n_media+1)),int(y_media/(n_media+1)))

def coords_ventana(x_c,y_c,tam=500):
    vnt=int(tam/2)
    xmin,xmax=x_c-vnt,x_c+vnt 
    ymin,ymax=y_c-vnt,y_c+vnt 
    if xmin<0:
        xmin=0
        xmax=tam
    if ymin<0:
        ymin=0
        ymax=tam
    return (xmin,xmax,ymin,ymax)

def crop_embyo(frame,tam=500):
    n,x,y,x0,y0=0,0,0,1,1
    gray = cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    closing=cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3)))
    while n<4 and (x!=x0 or y!=y0):
        if n==0:
            x,y = centroide(closing)
        else:
            x0,y0=x,y
            xmin,xmax,ymin,ymax=coords_ventana(x,y,tam)
            x,y = centroide(closing[ymin:ymax, xmin:xmax])
            x= xmin + x
            y= ymin + y
        n+=1
    return coords_ventana(x,y,tam)




    