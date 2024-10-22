# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def transform_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image_aux = image
    return my_transforms(image_aux).unsqueeze(0).to(device)

def prediction(img,model):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
  img=Image.fromarray(img)
  image = transforms.Resize((480,480))(img)
  tensor = transform_image(image=image)
  model.to(device)
  with torch.no_grad():
      outputs = model(tensor)

  outputs = torch.argmax(outputs,1)
  mask = np.array(outputs.cpu())
  mask[mask==1]=75
  mask[mask==2]=255
  mask[mask==3]=150
  mask=np.reshape(mask,(480,480))
  return np.array(transforms.Resize(img.size)(Image.fromarray(mask.astype('uint8'))))

