# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:24:36 2021

@author: 109598034 luo
"""

import time
import os
import json

import numpy as np
from PIL import Image
import tensorflow as tf

import heapq

from numpy import dot
from numpy.linalg import norm

import cv2


#global variable
model_path = "new_mobile_model.tflite"
label_path = "class_labels.txt"
num_threads = None

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


#載入標籤檔
def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

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

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    
    top_k = results.argsort()[-19:][::-1]
    labels = load_labels(label_path)
    
    return results
    
    for i in top_k:
        if floating_model:
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            
        else:
          print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
        
          print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
          
if __name__ == '__main__':
    

    
    #image_path為目標image的位址
    #getPredict("images/American Country/00c0ed18-9131-41f1-976e-52a8be0a0a21.jpg");    
    vector = getPredict("../result/" + "image.jpg");
    vector = vector.tolist()
    
    print(vector)
    
    file_dict = {}
    results = []
    
    i = 0;
    for fur_img in os.listdir("../image_vector/"):
        
        img_f = open("../image_vector/" + fur_img)
        image_json = json.load(img_f)
        
        image_vector = image_json['vector']
        file_name = image_json['fileName']
        
        file_dict[i] = file_name

        #向量夾角
        result = dot(image_vector,vector)/(norm(image_vector)*norm(vector))
    
        results.append(result)
        #file_dict[i] = room
        i += 1
        
    small_number = heapq.nlargest(5, results)
    
    print(small_number)
    
    #分開重複的key index
    small_index = []
    for t in small_number:
        index = results.index(t)

        small_index.append(index)
        results[index] = 0

    
    for index in small_index:
        print(file_dict[index])
        

        #顯示圖片
        img = cv2.imread('E:\\3D-FUTURE-model\\3D-FUTURE-model\\' + file_dict[index]+ '\\image.jpg')
        
        cv2.imshow('My Image', img)
        # 按下任意鍵則關閉所有視窗
        cv2.waitKey(0)
        cv2.destroyAllWindows()
