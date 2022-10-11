# IndoorSceneRecommendation

基於設計風格的家具推薦與擺設系統

## 摘要

本論文提出的系統可以幫使用者快速生成出適當的房間裝飾,不需要具備相關的設
計知識或是花時間選擇家具的組合,只要提供類似的房間圖片系統便會推薦出相關的家
具並根據房間結構進行擺放。系統的設計依照使用者提供的風格模板進行分析,利用
Detectron2 來進行語意分割(Semantic Segmentation),從而將模板中的家具切割出來,並
基於 VGG-16 實踐以圖搜圖的功能找出外觀和顏色相近的家具,再經過 3D-FUTURE 訓
練而成的風格分類網路進一步過濾出相同風格的結果。藉由現有的場景生成算法(CSSG)
按照使用者的房間類型重新擺放家具位置,為了讓使用者更容易檢視結果,系統同時也
提供以擴增實境(AR)的方式檢視場景。

## Demo

[https://youtu.be/P0PdWNGvCko](https://youtu.be/P0PdWNGvCko)


## 系統架構

![系統架構](Img/系統架構.jpg)

## 系統流程

整個推薦系統架設在以Flask建立的[Server](Server)上，包含圖片分割、以圖搜圖以及風格分類的功能。
 
![系統流程](Img/系統流程.jpg)



-------------------------------------------------------------------------------

[Dectctron2](https://github.com/facebookresearch/detectron2)

[Deep Learning of Binary Hash Codes for Fast Image Retrieval](https://www.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf)

[CSSG](https://github.com/amazon-research/indoor-scene-generation-eai/tree/main/IndoorSceneSynthesis/ConstraintStochasticIndoorSceneGeneration)

[ARFundation](https://github.com/Unity-Technologies/arfoundation-samples)


