from ultralytics import YOLO

# model = YOLO('yolov8x.pt')
model = YOLO("/home/ubuntu/GITHUG/ultralytics/YOLO8/runs/detect/middle/weights/best.pt")

model.predict('/home/ubuntu/DataSet/middle/test/images',save=True,imgsz=1280,conf=0.3)