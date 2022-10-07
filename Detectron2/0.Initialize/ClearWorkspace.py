# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:45:56 2022

@author: 109598034

清理WorkSpace多餘的垃圾
"""

import shutil
import os


#移除資料夾
deletePath = ['result', 'recommend']

for d in deletePath:
    try:
        shutil.rmtree('../' + d)
    except OSError as e:
        print(e)
    else:
        print("The directory is deleted successfully")
        
#移除檔案    
#deleteFile = ['input.jpg', 'vector.json']
deleteFile = ['vector.json']

for f in deleteFile:
    
    try:
        os.remove('../' + f)
    except OSError as e:
        print(e)
    else:
        print("File is deleted successfully")