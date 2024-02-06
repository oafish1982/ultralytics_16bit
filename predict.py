from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8x.yaml')  # build a new model from YAML
# model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
model = YOLO("/runs/detect/train7_2560/weights/best.pt")  # build from YAML and transfer weights
model.to(device="cpu")
# Train the model
model.predict(source='/home/ubuntu/DataSet/FULL_middle/testpng/images',save=True)