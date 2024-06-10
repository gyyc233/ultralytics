# need install onnx and openvino
# yolo ---> onnx model ---> openvino model

from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")

# Run inference
# openvino was fast than yolo8 pt
results = ov_model("https://ultralytics.com/images/bus.jpg")

results = model("https://ultralytics.com/images/bus.jpg")
