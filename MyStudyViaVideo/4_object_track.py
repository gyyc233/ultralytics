import os # 获取当前工作路径
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

# 我的小新pro14 运行会导致蓝屏...
class ObjectTrack:

    def __init__(self, capture_index):

        self.capture_index=capture_index
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model=self.load_model()
        self.confidence=[]
        self.class_ids=[]

    def load_model(self):
        model=YOLO("yolov8m.pt")
        model.fuse()

        return model
    
    def run_model(self,frame,vis,conf,save):
        # yolo支持 BoT-SORT 和 ByteTrack 两种跟踪算法
        # BoT-SORT - Use botsort.yaml to enable this tracker.
        # ByteTrack - Use bytetrack.yaml to enable this tracker.
        # The default tracker is BoT-SORT.

        #  bytetrack.yaml path: ultralytics/cfg/trackers
        result=self.model.track(source=frame, show=vis, tracker="bytetrack.yaml")

        return result

    def __call__(self):
        cap=cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, image= cap.read()
            assert ret

            results=self.run_model(image,True,0.8,False)

            if(cv2.waitKey(1) & 0xFF == ord("q")):
                break

        cap.release()
        cv2.destroyAllWindows()
        

if __name__=="__main__":
    model=ObjectTrack(0)
    model()
