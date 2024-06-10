from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml') # built a new model from YAML
model = YOLO('yolov8n.pt') # load a pretrained model (recommended for training) tuijia
# model = YOLO('yolov8n.yaml').load('yolov8n.pt') # build from YAML and transfer weights

# Train the model
# 训练轮数 epochs 300 每一轮都遍历整个训练数据集。训练的轮数越多，模型对数据的学习就越充分，但也增加了训练时间
# 每批数量 batch 每个批次中的图像数量;一般认为batch越大越好。因为我们的batch越大我们选择的这个batch中的图片更有可能代表整个数据集的分布
# 图像大小 imgsz 640表示图像的宽度和高度均为640像素；可以指定一个整数值表示图像的边长，也可以指定宽度和高度的组合
# 数据加载时的工作线程数 workers 在数据加载过程中，可以使用多个线程并行地加载数据.windows系统下需设置为0，否则会报错
# 这是因为在linux系统中可以使用多个子进程加载数据，而在windows系统中不能
model.train(data='coco128.yaml', epochs=300, imgsz=640,device=0, workers=0,batch=4)
