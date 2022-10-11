# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 20:55:39 2022

@author: 109598034
"""

import numpy as np

from flask import Flask, request, send_file
from flask_cors import CORS

import json

#1. 偵測房間物件
import torch
print(torch.__version__)
print(torch.cuda.is_available())



# Some basic setup:
# Setup detectron2 logger
import detectron2


# import some common libraries
import numpy as np
import os, json, cv2


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#2.分類器
import time
from PIL import Image
import tensorflow as tf

import shutil
import heapq

from numpy import dot
from numpy.linalg import norm



import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.preprocessing import image


#global variable
model_path = "E:\\3D-FUTURE-model\\new_mobile_model.tflite"
label_path = "E:\\3D-FUTURE-model\\class_labels.txt"
num_threads = None

datasetPath = 'E:/3D-FUTURE-model/3D-FUTURE-model/'

Path = "rooms/"


style = {
    'Modern':0,
    'Chinoiserie':1,
    'Kids':2,
    'European':3,
    'Japanese':4,
    'Southeast Asia':5,
    'Industrial':6,
    'American Country':7,
     'Vintage':8,
    'Light Luxury':9,
    'Mediterranean':10,
    'Korean':11,
    'New Chinese':12,
    'Nordic':13,
    'European Classic':14,
    'Others':15,
     'Ming Qing':16,
    'Neoclassical':17,
    'Minimalist':18
}

rule_index = {
        
     'Table':0,
     'Sofa':1,
     'Chair':2,
     'Bed':3
        
}

Type2CDF = {
        
    'Children Cabinet':'19',
    'Nightstand':'19',
    'Bookcase / jewelry Armoire':'19',
    'Wardrobe':'19',
    'Coffee Table':'148',
    'Corner/Side Table':'129',
    'Sideboard / Side Cabinet / Console Table':'129',
    'Wine Cabinet':'19',
    'TV Stand':'147',
    'Drawer Chest / Corner cabinet':'19',
    'Shelf':'126',
    'Round End Table':'26',
    'King-size Bed':'76',
    'Bunk Bed':'76',
    'Bed Frame':'76',
    'Single bed':'76',
    'Kids Bed':'76',
    'Dining Chair':'21',
    'Lounge Chair / Cafe Chair / Office Chair':'88',
    'Dressing Chair':'21',
    'Classic Chinese Chair':'88',
    'Barstool':'21',
    'Dressing Table':'129',
    'Dining Table':'26',
    'Desk':'128',
    'Three-Seat / Multi-seat Sofa':'115',
    'armchair':'88',
    'Loveseat Sofa':'115',
    'L-shaped Sofa':'115',
    'Lazy Sofa':'115',
    'Chaise Longue Sofa':'115',
    'Footstool / Sofastool / Bed End Stool / Stool':'155',
    'Pendant Lamp':'74',
    'Ceiling Lamp':'74'    
}

app = Flask(__name__)
CORS(app)


#載入模型(Detectron2)
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
predictor = DefaultPredictor(cfg)




vector = np.zeros(19)
recommend_furnitures = []

f = open('E:\\3D-FUTURE-model\\model_info.json')
model_info = json.load(f)

model_dict = {}
style_dict = {}

#把家具和種類的對應存成字典
for model in model_info:
    model_dict[model['model_id']] = model['category']
    style_dict[model['model_id']] = model['style']

    


#VGG16
class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'   # None代表随机初始化，即不加载预训练权重
        self.pooling = 'max'  # avg
        self.model_vgg = VGG16(weights=self.weight,
                               input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                               pooling=self.pooling,
                               include_top=False)
        # self.model_vgg.predict(np.zeros((1, 224, 224, 3)))
 
    # 提取vgg16最后一层卷积特征( Use vgg16/Resnet model to extract features Output normalized feature vector)
    def vgg_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = self.model_vgg.predict(img)
        # print(feat.shape)
        norm_feat = feat[0] / np.linalg.norm(feat[0])
        return norm_feat
    
def get_feature_from_hdf5(path):
    # read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File(path, 'r')
    feats = h5f['dataset_1'][:]
    names = h5f['dataset_2'][:]
    h5f.close()
    return feats, names

AnotherStyle = ['American Country',
                'Chinoiserie',
                'Industrial',
                'Japanese',
                'Korean',
                'Light Luxury',
                'Minimalist',
                'Modern',
                'Southeast Asia',
                'Vintage'
]
def copyImage(src, dst):
    
    # Copy the content of
    # source to destination
    shutil.copyfile(src, dst)
    
def getPredict(image_path):

    #tensorflow model interpreter
    interpreter = tf.lite.Interpreter(model_path, num_threads)
    interpreter.allocate_tensors()
    
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    floating_model = input_details[0]['dtype'] == np.float32
    
    
      # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(image_path).resize((width, height))
    
    
    # add N dim
    input_data = np.expand_dims(img, axis=0)  
    
    
    if floating_model:
        input_data = (np.float32(input_data) - 0) / 255
        
        
    interpreter.set_tensor(input_details[0]['index'], input_data)


    interpreter.invoke()

    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    
    top_k = results.argsort()[-10:][::-1]
    labels = load_labels(label_path)

    return labels[top_k[0]]
    
def class10(image_path='22.png', maxres=3, wantStyle = 'currentStyle', Type = 'Bed'):
    path = 'E:\\3D-FUTURE-scene\\CataTrain\\' + Type + '.h5'
   
    feats, names = get_feature_from_hdf5(path)
    
    wantStyle = getPredict(image_path)
    print("want style", wantStyle)
    
    # init VGGNet16 model
    model = VGGNet()  
 
    # extract query image's feature, compute simlarity score and sort
    img_feat = model.vgg_extract_feat(image_path)  # 修改此处改变提取特征的网络
    scores = np.sqrt(np.dot(img_feat, feats.T))
    #scores = np.dot(img_feat, feats.T)/(np.linalg.norm(img_feat)*np.linalg.norm(feats.T))
    #scores = np.sqrt(sum(pow(img_feat-feats.T,2)))
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    imlist = []
    hit = False
    #for i, index in enumerate(rank_ID[0:maxres]):
    for i, index in enumerate(rank_ID[0:100]):

        """
        style = str(names[index]).split(".")[0].split("'")[1].strip().split('_')[0]
        
        if style not in AnotherStyle:
            style = model_dict[style]
            obj_name = str(names[index]).split(".")[0].split("'")[1].split('_')[0]
        else:
            obj_name = str(names[index]).split(".")[0].split("'")[1].split('_')[2]
       """
       
        obj_name = str(names[index]).split(".")[0].split("'")[1].split('_')[0]
        furniture_type = model_dict[obj_name]
        style = style_dict[obj_name]
            
        if style == 'Vintage/Retro':
            style = 'Vintage'
        
        if style == 'European Classic':
            style = 'European'
            
        if style == 'Ming Qing':
            style = 'Chinoiserie'  
            
        if style == 'New Chinese':
            style = 'Chinoiserie'    
        
        if obj_name == image_path.split('\\')[-1].split('_')[0]:
            hit = True
            
        #確保風格
        #if rank_score[i] >= 0.85 or wantStyle == style:
        if rank_score[i] >= 0.5:
            if obj_name not in imlist:
                imlist.append(obj_name)
                print("info", obj_name, furniture_type, style)
        if len(imlist) == maxres:
            break

    #寫入結果
    global recommend_furnitures
    for im in imlist:
        im = getFurnitureType(im)
        recommend_furnitures.append(im)
        
    
    if len(imlist) == 0:
        return [style, "null", "null", False]
    top_1_score = rank_score[0]
    top_1_md5 = imlist[0]

    style = model_dict[top_1_md5]

    return [style, top_1_md5, top_1_score, hit]
    
    


#小工具
def crop_object(image, box):
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
  return crop_img

def getFurnitureType(name):
    
    global model_dict
    return Type2CDF[model_dict[name]] + '_' + name


#get class name
catelog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
AcceptType = ['sofa', 'table', 'chair', 'bed', 'cabinet','stool', 'dresser', 'drawer']    


def recommend(image, room_type):
    
    #initilize global
    global vector
    vector = np.zeros(19)
    global recommend_furnitures
    recommend_furnitures = []
    
    AnotherType = ['table', 'cabinet','stool', 'chair', 'sofa', 'bed']
    
    #移除資料夾
    deletePath = ['result', 'recommend']

    for d in deletePath:
        try:
            shutil.rmtree(d)
        except OSError as e:
            print(e)
        else:
            print("The directory is deleted successfully")

    
    outputs = predictor(image)
    
    
    #分割物件的segment
    boxes = outputs["instances"].pred_boxes
    labels = outputs["instances"].pred_classes
    
    os.makedirs('result')
    #瀏覽每一個BoundingBox
    for i in range(0, len(boxes)):

            
        for act in AcceptType:
            print("label", catelog[labels[i]])
            if act in catelog[labels[i]]:
                
                if act == 'dresser':
                    act = 'cabinet'
                if act == 'drawer':
                    act = 'stool'
                    
                if act in AnotherType:
                    AnotherType.remove(act)
                box = list(boxes)[i].detach().cpu().numpy()
                crop_img = crop_object(image, box)
                cv2.imwrite('result\\' + act.title() + "_" + str(i) +'.jpg', crop_img)
                break

        continue
        box = list(boxes)[i].detach().cpu().numpy()
        crop_img = crop_object(image, box)
        cv2.imwrite('result\\' + catelog[labels[i]] + str(i) +'.jpg', crop_img)
        #print("label", catelog[labels[i]])

    
    #將結果進行比對
    for img in os.listdir('result'):
        
        Type = img.split('_')[0]
        info = class10(os.path.join('result', img), maxres=5, Type = Type)
        
        print(info, img)
    
    #補正
    for another in AnotherType:
        print("another", AnotherType)
        info = class10('input.jpg', maxres=5, Type = another)

#載入標籤檔
def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


#預測資料
@app.route("/predict", methods=['GET', 'POST'])
def predict():

	#傳入資料
    img = request.files.get('image')
    img.save("input.jpg")

    room_type = request.values['room_type']
    
    #預測
    im = cv2.imread("input.jpg")
    recommend(im, room_type)

    #輸出成json 字串
    json_dict = {}
    json_dict['furniture'] = recommend_furnitures
    
    json_dict = json.dumps(json_dict)
    return json_dict


#下載圖片
@app.route('/get_image', methods=['GET', 'POST'])
def get_image():
    filename = request.values['fileName']
    imgPath = datasetPath + filename + "/image.jpg"
    return send_file(imgPath, mimetype='image/jpg')

#下載模型
    
@app.route('/get_obj_file', methods=['GET', 'POST'])
def get_obj_file():
    filename = request.values['fileName']
    objPath = datasetPath + filename + "/raw_model.obj"
    return send_file(objPath)

@app.route('/get_obj_texture', methods=['GET', 'POST'])
def get_obj_texture():
    filename = request.values['fileName']
    objPath = datasetPath + filename + "/texture.png"
    return send_file(objPath, mimetype='image/png')

@app.route("/", methods=['GET'])
def index():
    
    return "Hello"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8090, debug=False)