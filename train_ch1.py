from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8x.yaml')  # build a new model from YAML
# model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x.yaml')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/ubuntu/GITHUG/ultralytics/YOLO8/FULL_middle.yaml', workers=2,
                      epochs=600, imgsz=1280, device=0, optimizer="Adam", lr0=0.001, hsv_h=0.0, hsv_s=0.0,
                      hsv_v=0.0, batch=4)
