from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image
import cv2
import numpy as np
# Download YOLOv8 model
yolov8_model_path = "/home/ubuntu/GITHUG/ultralytics/YOLO8/runs/detect/middle/weights/best.pt"

# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     model_path=yolov8_model_path,
#     confidence_threshold=0.1,
#     device="cuda:1",  # or 'cuda:0'
# )

# # With an image path
# result = get_prediction("/home/ubuntu/DataSet/middle/test/images/AS-0.46-1 (8).png", detection_model)

# # With a numpy image
# result = get_prediction(read_image("/home/ubuntu/DataSet/middle/test/images/AS-0.46-1 (8).png"), detection_model)

# result.export_visuals(export_dir="demo_data/")
# Image("demo_data/prediction_visual1.png")


# im = cv2.imread("/home/ubuntu/DataSet/middle/test/images/AS-0.46-1 (8).png", cv2.IMREAD_UNCHANGED)  # BGR
# if im.dtype == np.uint16:
#     # 假设需要将16位灰度图转换为三通道的16位图
#     im = np.stack([im, im, im], axis=-1)

# result = get_sliced_prediction(
#     "/home/ubuntu/DataSet/middle/test/images/AS-0.46-1 (8).png",
#     detection_model,
#     slice_height=256,
#     slice_width=256,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2
# )


# # Access the object prediction list
# object_prediction_list = result.object_prediction_list

# # Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
# result.to_coco_annotations()[:3]
# result.to_coco_predictions(image_id=1)[:3]
# result.to_imantics_annotations()[:3]
# result.to_fiftyone_detections()[:3]

predict(
    model_type="yolov8",
    model_path=yolov8_model_path,
    model_device="cuda:1",  # or 'cuda:0'
    model_confidence_threshold=0.5,
    source="/home/ubuntu/DataSet/middle/test/images",
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

