#python 3.8.3 conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch  detectron2 0.6

"""

Input:Indoor Scene
Output:將圖片中的家具切割到..\\result資料夾

"""

import torch
import torchvision

print(torch.__version__)
print(torch.cuda.is_available())



# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.data import MetadataCatalog


# import some common libraries
import numpy as np
import os, json, cv2, random


from PIL import Image




# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def crop_object(image, box, mask):
  """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
  """

  x = box[0]
  y = box[1]
  
  w = box[2] - box[0]
  h = box[3] - box[1]
  
  crop_img = image[int(y):int(y+h), int(x):int(x+w)]
  
  # Crop the mask to match the cropped image
  cropped_mask = mask.crop((int(x), int(y), int(x + w), int(y + h)))
  background = Image.new('RGB', (cropped_mask.size),(255, 255, 255))

  composite = Image.composite(Image.fromarray(crop_img), background, cropped_mask)
  

  
  #return crop_img
  return np.array(composite)

im = cv2.imread("..\\input.jpg")


cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


#get class name
labels = outputs["instances"].pred_classes
catelog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
print(catelog)
AcceptType = ['sofa, table chair', 'bed', 'cabinet', 'dresser']

# Get pred_boxes from Detectron2 prediction outputs
boxes = outputs["instances"].pred_boxes
# Select 1 box:
#box = list(boxes)[0].detach().cpu().numpy()
# Crop the PIL image using predicted box coordinates
#crop_img = crop_object(im, box)



if not os.path.isdir("../result"):
    os.makedirs("../result")

# 寫入圖檔
    
# Get the masks
masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))

# Pick an item to mask


i = 0
for i in range(0, len(boxes)):
    """
    for act in AcceptType:
        
        if catelog[labels[i]] in act:
            box = list(boxes)[i].detach().cpu().numpy()
            crop_img = crop_object(im, box)
            cv2.imwrite('..\\result\\' + catelog[labels[i]].title() + '_' + str(i) +'.jpg', crop_img)
            break
    """
    # Pick an item to mask
    item_mask = masks[i]
    # Create a PIL image out of the mask
    mask = Image.fromarray((item_mask * 255).astype('uint8'))
    
    box = list(boxes)[i].detach().cpu().numpy()
    crop_img = crop_object(im, box, mask)


    cv2.imwrite('..\\result\\' + catelog[labels[i]] + str(i) +'.jpg', crop_img)
    
    
#顯示分割結果
"""
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.namedWindow("title",0);
cv2.resizeWindow("title", 640, 480);
cv2.imshow("title", out.get_image()[:, :, ::-1])
cv2.waitKey(0) 
"""

