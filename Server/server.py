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



#global variable
model_path = "2.CaculateVector/new_mobile_model.tflite"
label_path = "2.CaculateVector/class_labels.txt"
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


#載入模型
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

#把家具和種類的對應存成字典
for model in model_info:
    model_dict[model['model_id']] = model['category']


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
    


def recommend(image, room_type):
    
    #initilize global
    global vector
    vector = np.zeros(19)
    global recommend_furnitures
    recommend_furnitures = []

    
    outputs = predictor(image)
    
    #分割物件的segment
    boxes = outputs["instances"].pred_boxes
    
    #加總向量
    for i in range(0, len(boxes)):
        box = list(boxes)[i].detach().cpu().numpy()
        crop_img = crop_object(image, box)
        getPredict(crop_img)
    
    vector = vector / len(boxes)
                          
    print(vector)
    #比對最接近的前N個場景
    
    results = []
    file_dict = {}
    i = 0
    for room in os.listdir(Path + room_type):
        
        room_f= open(Path + room_type + "/" + room)
        room_vector = json.load(room_f)['vector']
    
        
        result = dot(room_vector,vector)/(norm(room_vector)*norm(vector))
    
        results.append(result)
        file_dict[i] = room
        i += 1

    #sort n smallest result
    #向量之間夾角越小，表示兩個向量的方向越接近，方向一致夾角度數θ為0，餘弦值為1；反之向量之間夾角θ越大，表示兩個向量的方向差異越大，當方向相反時餘弦值為-1。
    small_number = heapq.nlargest(5, results)
    
    #分開重複的key index
    small_index = []
    for t in small_number:
        index = results.index(t)
        small_index.append(index)
        results[index] = 0
    
    #把最小的圖片列出來
    for index in small_index:
        room = open(Path + room_type + "/" + file_dict[index])
        furniture = json.load(room)['furniture']
    
        for f in furniture:
            """
            src = datasetPath + f + "/image.jpg"
            dst = "../recommend/" + f + ".jpg"
            """
            f = getFurnitureType(f)
            if f not in recommend_furnitures:
                recommend_furnitures.append(f)
                
                
def findSimiliaryImg(fur_type, number):
    
    #已經被加進來的家具
    global recommend_furnitures
    
    #input場景的向量
    global vector
    
    results = []
    file_dict = {}
    
    i = 0
    for furnitureInfo in os.listdir('image_vector'):
        fur_f= open('image_vector/' + furnitureInfo)
        
        fur_info = json.load(fur_f)
        
        if fur_info['type'] == fur_type:
            
            result = dot(fur_info['vector'],vector)/(norm(fur_info['vector'])*norm(vector))
            results.append(result)
            file_dict[i] = fur_info['fileName']
            i += 1
    
     #sort n smallest result
    #向量之間夾角越小，表示兩個向量的方向越接近，方向一致夾角度數θ為0，餘弦值為1；反之向量之間夾角θ越大，表示兩個向量的方向差異越大，當方向相反時餘弦值為-1。
    small_number = heapq.nlargest(number, results)
    
    #分開重複的key index
    small_index = []
    for t in small_number:
        index = results.index(t)
        small_index.append(index)
        results[index] = 0   
        
        
    for index in small_index:
        recommend_furnitures.append(file_dict[index])
        print(file_dict[index])

        
def getPredict(crop_img):

    #Translate to PIL
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    crop_img = Image.fromarray(crop_img)
    
    #tensorflow model interpreter
    interpreter = tf.lite.Interpreter(model_path, num_threads)
    interpreter.allocate_tensors()
    
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    floating_model = input_details[0]['dtype'] == np.float32
    
    
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    img = crop_img.resize((width, height))
    
    
    # add N dim
    input_data = np.expand_dims(img, axis=0)  
    
    
    if floating_model:
        input_data = (np.float32(input_data) - 0) / 255
        
        
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    
    top_k = results.argsort()[-19:][::-1]
    labels = load_labels(label_path)
    
    for i in top_k:
        if floating_model:
            #print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            global vector
            vector[style[labels[i]]] += float(results[i])
        else:
          print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
        
          print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))  

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
    rules = request.values['rules']

    #預測
    im = cv2.imread("input.jpg")
    recommend(im, room_type)
    
    json_rule = json.loads(rules)
    
    if len(json_rule['rules']) != 0:
        rules_array = [0, 0, 0, 0]
        
        for rule in json_rule['rules']:
            rules_array[ rule_index[rule['furniture']] ] += rule['count']

        #檢查類別有沒有被滿足
        for r in recommend_furnitures:
            f = open('image_vector\\' + str(r) + ".json")
            model_info = json.load(f)
            print(model_info['type'])
            
            if model_info['type'] in rule_index:
                rules_array[rule_index[model_info['type']]] -= 1
        
        for item in rule_index.keys():
            if rules_array[rule_index[item]] > 0:
                print('我要補正這個結果')
                
                findSimiliaryImg(item, rules_array[rule_index[item]])
    
    
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