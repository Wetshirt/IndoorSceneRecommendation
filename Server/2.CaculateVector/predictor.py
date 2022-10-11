# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:24:36 2021

@author: 109598034 luo

Input:一練串家具的照片
Output:加總所有的向量取平均值
"""

import time
import os
import json

import numpy as np
from PIL import Image
import tensorflow as tf


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

vector = np.zeros(19)


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
    
    for i in top_k:
        if floating_model:
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            vector[style[labels[i]]] += float(results[i])
        else:
          print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
        
          print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
          
if __name__ == '__main__':
    
    #image_path為目標image的位址
    #getPredict("images/American Country/00c0ed18-9131-41f1-976e-52a8be0a0a21.jpg");    
    for f in os.listdir("../result"):
        getPredict("../result/" + f);
        print("Next")
    print('finish')
    vector /= len(os.listdir("../result"))

    result = vector.tolist()
    
    output_file = "../vector.json"
    with open(output_file, "w") as output_file:
        json.dump(result , output_file, indent = 4)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    