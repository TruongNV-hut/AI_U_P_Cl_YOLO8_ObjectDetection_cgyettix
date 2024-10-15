from ultralytics import YOLO

model = YOLO("pretrain/yolov8n.pt")

result = model.train(data='data/mydataset.yaml', epochs=10)
