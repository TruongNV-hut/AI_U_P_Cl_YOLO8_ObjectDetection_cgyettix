from ultralytics import YOLO

model = YOLO("./pretrain/yolov8n.pt")  # load a pretrained YOLOv8n model
model.export(format="tflite",opset=13)  # export the model to ONNX forma