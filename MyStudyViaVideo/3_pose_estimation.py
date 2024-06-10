import os # 获取当前工作路径
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class PoseEstimation:

    def __init__(self, capture_index):

        self.capture_index=capture_index
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model=self.load_model()
        self.confidence=[]
        self.class_ids=[]

    def load_model(self):
        model=YOLO("yolov8n-pose.pt")
        model.fuse()

        return model
    
    def run_model(self,frame,vis,conf,save):
        result=self.model(source=frame,show=vis,conf=conf,save=save)

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

            if(cv2.waitKey(10)==27):
                break

        cap.release()
        cv2.destroyAllWindows()
        

if __name__=="__main__":
    model=PoseEstimation(0)
    model()
