- [YOLOv8's Val mode is advantageous](#yolov8s-val-mode-is-advantageous)
- [Key Features of Val Mode](#key-features-of-val-mode)
- [run](#run)

## YOLOv8's Val mode is advantageous

- Precision: Get accurate metrics [精准指标] like mAP50, mAP75, and mAP50-95 to comprehensively evaluate your model.获取mAP50、mAP75 和 mAP50-95 等精确指标，全面评估模型
- Convenience: Utilize built-in features that remember training settings, simplifying the validation process.
- Flexibility: Validate your model with the same or different datasets and image sizes.
- Hyperparameter Tuning: Use validation metrics to fine-tune your model for better performance.使用 validation metrics 微调模型，以获得更好性能


## Key Features of Val Mode

These are the notable functionalities offered by YOLOv8's Val mode:

- Automated Settings: Models remember their training configurations for straightforward validation. 模型会记住它们的训练配置，以便直接验证
- Multi-Metric Support: Evaluate your model based on a range of accuracy metrics.
- CLI and Python API: Choose from command-line interface or Python API based on your preference for validation.
- Data Compatibility: Works seamlessly with datasets used during the training phase as well as custom datasets. 数据兼容性：与训练阶段使用的数据集以及自定义数据集无缝协作。

**YOLOv8 models automatically remember their training settings, so you can validate a model at the same image size and on the original dataset easily with just `yolo val model=yolov8n.pt` or `model('yolov8n.pt').val()`
**

## run

```python
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
```

output 

```bash
Ultralytics YOLOv8.1.0 🚀 Python-3.9.19 torch-1.10.0+cu102 CUDA:0 (GeForce MX250, 2048MiB)
YOLOv8n summary (fused): 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
val: Scanning D:\datasets\coco8\labels\val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:00<?, 
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<0
                   all          4         17          1      0.783      0.888      0.738
                person          4         10          1        0.2        0.6      0.457
                   dog          4          1          1          1      0.995      0.796
                 horse          4          2          1          1      0.995      0.784
              elephant          4          2          1        0.5       0.75        0.6
              umbrella          4          1          1          1      0.995      0.895
          potted plant          4          1          1          1      0.995      0.895
Speed: 1.3ms preprocess, 24.1ms inference, 0.0ms loss, 2.5ms postprocess per image
Results saved to D:\MyGithubCode\runs\detect\val20

```
