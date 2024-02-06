
from ultralytics import settings
from ultralytics import YOLO


print(settings)
settings.update({"tensorboard":True})
# Load a model
# model = YOLO('yolov8x.yaml')  # build a new model from YAML
# model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x.yaml')  # build from YAML and transfer weights
# model =YOLO("/home/ubuntu/GITHUG/ultralytics/runs/detect/train6_SGD/weights/best.pt")
# Train the model
results = model.train(data='/home/ubuntu/GITHUG/ultralytics/YOLO8/FULL_middle_png.yaml', workers=1,
                      epochs=300, imgsz=1280, device=[0,1], optimizer="Adam", lr0=0.001, hsv_h=0.0, hsv_s=0.0,
                      hsv_v=0.0, batch=4, amp=True,patience=100)
# mosaic=0,copy_paste=0.0,mixup=0
