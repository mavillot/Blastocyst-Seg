# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '../utils/')
from kfold import *
from metrics import *
from sklearn.model_selection import KFold
import random

number_of_the_seed = 2020

random.seed(number_of_the_seed)
set_seed(number_of_the_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path=Path('../dataset/')
path_train=path/'train'
path_model=Path('../models')

transforms=Compose([HorizontalFlip(p=0.5),Rotate(p=0.40,limit=10)],p=1)
transformPipeline=SegmentationAlbumentationsTransform(transforms)
codes = np.array(['Background','ZP', 'TE', 'ICM'])

bs = 2
callbacks = [EarlyStoppingCallback(patience=3),SaveModelCallback(fname='model'),ReduceLROnPlateau(patience=3)]
opt = ranger

get_y_fn = lambda x: Path(str(x).replace("train","GT_Mask"))

img = PILImage.create(path_train/'Blast_PCRM_1201754 D5.BMP')
img = transform_image(img)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

k=1
metricas={'unet':{'zp':[],'te':[],'icm':[]},
          'hrnet':{'zp':[],'te':[],'icm':[]},
          'deeplab':{'zp':[],'te':[],'icm':[]}}

gtruth_zp=[get_mask(f,'ZP',path) for f in files]
gtruth_te=[get_mask(f,'TE',path) for f in files]
gtruth_icm=[get_mask(f,'ICM',path) for f in files]
          
for indx_train, indx_valid in kfold.split(get_files(path_train)):
    trainDB = datablock(indx_valid)
    trainDLS = trainDB.dataloaders(path_train,bs=bs)
    unet = unet_learner(trainDLS, resnet34, self_attention=True, act_cls=Mish, opt_func=opt,metrics=[DiceMulti()],
                        cbs=callbacks)
    hrnet = get_segmentation_learner(dls=trainDLS, number_classes=4, segmentation_type="Semantic Segmentation",
                                 architecture_name="hrnet", backbone_name="hrnet_w30", 
                                 metrics=[background,zp,te,icm,msa],wd=1e-2,
                                 pretrained=True,normalize=True).to_fp16()
    deeplab = get_segmentation_learner(dls=trainDLS, number_classes=4, segmentation_type="Semantic Segmentation",
                                 architecture_name="deeplabv3+", backbone_name="resnet50", 
                                 metrics=[background,zp,te,icm,msa],wd=1e-2,
                                 pretrained=True,normalize=True).to_fp16()
    #UNET
    lr_steep=unet.lr_find()
    unet.fit_one_cycle(20,lr_steep)
    unet.fit_one_cycle(20,lr_steep)
    unet.unfreeze()
    lr_steep=unet.lr_find()
    unet.fit_one_cycle(5,lr_steep)
    metricas
    
    aux=unet.model
    aux=aux.cpu()
    img=img.cpu()
    traced_cell=torch.jit.trace(aux, (img))
    traced_cell.save(str(path_model)+f"/kfold_unet/{k}.pth")
    
    #HRNET
    lr_steep=hrnet.lr_find()
    hrnet.fit_one_cycle(20,lr_steep)
    hrnet.fit_one_cycle(20,lr_steep)
    hrnet.unfreeze()
    lr_steep=hrnet.lr_find()
    hrnet.fit_one_cycle(5,lr_steep)
    aux=hrnet.model
    aux=aux.cpu()
    img=img.cpu()
    traced_cell=torch.jit.trace(aux, (img))
    traced_cell.save(str(path_model)+f"/kfold_hrnet/{k}.pth")
    
    #DEEPLAB
    lr_steep=learn.lr_find()
    deeplab.fit_one_cycle(20,lr_steep)
    deeplab.fit_one_cycle(20,lr_steep)
    deeplab.unfreeze()
    lr_steep=deeplab.lr_find()
    deeplab.fit_one_cycle(5,lr_steep)
    aux=deeplab.model
    aux=aux.cpu()
    img=img.cpu()
    traced_cell=torch.jit.trace(aux, (img))
    traced_cell.save(str(path_model)+f"/kfold_deeplab/{k}.pth")
    k+=1

