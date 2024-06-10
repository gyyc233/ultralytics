from ultralytics import YOLO

# Load a model
model = YOLO('D:/MyGithubCode/runs/detect/train23/weights/best.pt')  # load a custom model 加载自定义模型

# Validate the model
metrics = model.val(data='coco8.yaml',
                               imgsz=640,
                               batch=4,
                               conf=0.6,
                               iou=0.6,
                               device='0',workers=0)

# Validate the model
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
