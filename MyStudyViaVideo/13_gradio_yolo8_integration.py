# gradio 为AI模型的使用提供了快速简易的可视化界面
# gradio & yolov8 integration
# 运行程序后打开对应端口即可

import gradio as gr

from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
    )
    return results[0].plot() if results else None


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio YOLO8",
    description="Upload images for YOLO11 object detection.",
)
iface.launch()

