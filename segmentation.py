# -*- coding: utf-8 -*-
import torch
import cv2
import sys

from utils.crop import crop_embyo
from utils.preds import prediction


if len(sys.argv) > 2:
    path_img = sys.argv[1]
    path_model = sys.argv[2]
else:
    print("No values provided.")

# CAMBIAR PARA INTRODUCIR 
#path_img="img.png"
#path_model='models/models/hrnet.pth'


# Load the trained model
model = torch.jit.load(path_model)
model = model.cpu()


img=cv2.imread(path_img)
xmin,xmax,ymin,ymax=crop_embyo(img,tam=500)
img=img[ymin:ymax, xmin:xmax]
mask=prediction(img,model)
cv2.imwrite('prediction.png',mask)