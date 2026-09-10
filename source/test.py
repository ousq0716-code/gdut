from ultralytics import YOLO


model = YOLO("yolov8n.pt")


results = model("E:/Ctrain/jpgtrain/images.jpg", save=True) # 测试图像 可自己找一张换成自己的路径
results[0].show()
path = model.export(format="onnx")