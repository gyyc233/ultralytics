from ultralytics.utils.benchmarks import benchmark

benchmark("yolov8n.pt",data="coco8.yaml",imgsz=640,half=False,int8=True,device="cpu")