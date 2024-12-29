import cv2
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation

model = YOLO("yolov8m.pt")
names = model.model.names
cap = cv2.VideoCapture("D:/work_space/MyTestData/cara.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

line_pts = [[0,h//2],[w,h//2]]

speed_obj = speed_estimation.SpeedEstimator();
speed_obj.set_args(reg_pts=line_pts, names=names,view_img=True)

while cap.isOpened():
    success,im0=cap.read()
    if not success:
        print("video frame is empty or video processing has been successfully completed")
        break

    tracks=model.track(im0,persist=True,show=False)

    im0=speed_obj.estimate_speed(im0,tracks)
    out.write(im0)

cap.release()
out.release()
cv2.destroyAllWindows()

