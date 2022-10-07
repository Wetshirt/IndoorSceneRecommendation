set root=C:\Users\109598034\anaconda3

call %root%\Scripts\activate.bat %root%

call conda activate tensorflow

cd 1.Segmentation
python FurnitureDetecter.py

cd ../2.CaculateVector
python predictor.py

cd ../3.CompareScene
python FindSimilarRoom.py

pause
