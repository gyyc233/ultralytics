import os # 获取当前工作路径
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index=capture_index
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model=self.load_model()
        self.confidence=[]
        self.class_ids=[]

    def load_model(self):
        # model=YOLO('D:/MyGithubCode/runs/detect/train23/weights/best.pt')

        model = YOLO("yolov8n-seg.yaml")
        model=YOLO("yolov8n-seg.pt")
        model.fuse() # 融合 pytorch 二维卷积和归一化处理

        return model
    
    def run_model(self,frame,vis,conf,save):
        result=self.model(source=frame,show=vis,conf=conf,save=save)

        return result

    def plot_bboxes(self, results, frame):

        xyxys=[]

        # extract detections for person class  提取obb识别结果并将其可视化
        for result in results:
            boxes=result.boxes.cpu().numpy()
            xyxys=boxes.xyxy

            print("==================================")
            print("conf: ",boxes.conf)
            print("cls: ",boxes.cls)
            print("id: ",boxes.id)
            print("xyxy: ",boxes.xyxy)
            print("xywh: ",boxes.xywh)
            print("xyxyn: ",boxes.xyxyn)
            print("xywhn: ",boxes.xywhn)

            # save result
            # self.confidence.append(boxes.conf)
            # self.class_ids.append(boxes.cls)
            
            for xyxy in xyxys:
                cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])),(0,255,0))

        return frame

    def __call__(self):
        cap=cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, image= cap.read()
            assert ret

            results=self.run_model(image,False,0.8,False)
            image=self.plot_bboxes(results,image)

            cv2.imshow('YOLOv8 Detection', image)

            if(cv2.waitKey(10)==27):
                break

        cap.release()
        cv2.destroyAllWindows()

    def predict_image(self, frame): 
        print("predict image:",frame)
        source = cv2.imread(frame)
        results = self.model.predict(source)

        for r in results:
            print("mask: ", r.masks)

        return results
        

if __name__=="__main__":
    model=ObjectDetection(0)
    model()

    # predict
    model.predict_image(os.getcwd()+"\\bus.jpg")


