- [Use YOLO pre-configured model](#use-yolo-pre-configured-model)
- [Usage](#usage)

## Use YOLO pre-configured model

- directory: `ultralytics/ultralytics/cfg/models/v8
- 可以直接使用yolo训练好的模型，也可以基于此自定义自己的模型

## Usage

CLI

```bash
# Train a YOLOv8n model using the coco8 dataset for 100 epochs
yolo task=detect mode=train model=yolov8n.yaml data=coco8.yaml epochs=100
```

Python environment

```py
from ultralytics import YOLO

# Initialize a YOLOv8n model from a YAML configuration file
model = YOLO("model.yaml")

# If a pre-trained model is available, use it instead
# model = YOLO("model.pt")

# Display model information
model.info()

# Train the model using the COCO8 dataset for 100 epochs
model.train(data="coco8.yaml", epochs=100)
```
