# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:21:58 2022

@author: 109598034

Input:特徵向量
Output:依據向量找出所有場景向量夾角最小的5個場景，並列出裡面出現過的家具
"""
import json
import shutil
import os
import heapq

from numpy import dot
from numpy.linalg import norm

datasetPath = 'E:/3D-FUTURE-model/3D-FUTURE-model/'

Path = "../rooms/"
room_type = "LivingRoom"

f = open("../vector.json")
vector = json.load(f)

results = []
file_dict = {}
i = 0
for room in os.listdir(Path + room_type):
    
    room_f= open(Path + room_type + "/" + room)
    room_vector = json.load(room_f)['vector']

    #向量夾角
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

#把結果寫入recommend
if not os.path.isdir("../recommend"):
    os.makedirs("../recommend")

for index in small_index:
    room = open(Path + room_type + "/" + file_dict[index])
    furniture = json.load(room)['furniture']

    for f in furniture:
        src = datasetPath + f + "/image.jpg"
        dst = "../recommend/" + f + ".jpg"
        shutil.copyfile(src, dst)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        