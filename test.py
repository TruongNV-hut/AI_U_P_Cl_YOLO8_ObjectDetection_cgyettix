from ultralytics import YOLO
import cv2
from datetime import datetime

model = YOLO('pretrain/yolov8n.pt')
img_path = 'data/test/maybay.jpg'

timenow = datetime.now()
print(timenow)
results = model(show=False, source=img_path, save=True, save_txt=True, project="output/predicts", name='20240614', save_crop=True) 

timenow = datetime.now()
print(timenow)