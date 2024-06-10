# 使用预训练模型进行目标检测

from ultralytics import YOLO

# load a pretrained yolo8n model
# model=YOLO('yolov8n.pt')

# use custom model based pretrained model
model=YOLO('D:/MyGithubCode/runs/detect/train23/weights/best.pt')

# run inference on the source 进行推理
result=model(source=0,show=True, conf=0.4,save=True)
