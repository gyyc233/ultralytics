from ultralytics import YOLO


# detect
model=YOLO('yolov8m.pt')

result=model("bus.jpg",show=True, conf=0.4,save=True)

# segment
model=YOLO('yolov8m-seg.pt')

result=model("bus.jpg",show=True, conf=0.4,save=True)

# classic
model=YOLO('yolov8m-cls.pt')

result=model("bus.jpg",show=True, conf=0.4,save=True)

# pose
model=YOLO('yolov8m-pose.pt')

result=model("bus.jpg",show=True, conf=0.4,save=True)
