import cv2
from ultralytics import YOLO
from ultralytics.solutions import distance_calculation

model = YOLO("yolov8m.pt")
names = model.names
cap = cv2.VideoCapture("D:/work_space/MyTestData/cars.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# Init distance_calculation
# 这里检测的只是像素距离，没有结合相机内参与外参
dist_obj = distance_calculation.DistanceCalculation()
dist_obj.set_args(names=names,view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0,persist=True, show=False)
    dist_obj.start_process(im0,tracks)

cap.release()
