# https://www.bilibili.com/video/BV1mH4y1N7WH/?spm_id_from=333.999.0.0&vd_source=4d02a316606ea19e315b11bab27432aa

import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("D:/work_space/MyTestData/cars.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
# region_points = [(400, 0), (400, 400)]  # For line counting
region_points = [ (600, 200),(300, 200),(300, 430),(600, 430)]  # For rectangle region counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting

# Init ObjectCounter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True, reg_pts=region_points, classes_names=model.names, draw_tracks=True)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0,persist=True, show=False)
    counter.start_counting(im0, tracks)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
