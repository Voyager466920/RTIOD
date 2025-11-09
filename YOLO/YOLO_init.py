import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    model.train(data="coco8.yaml", epochs=30, workers=0, device=0)
    model.val(workers=0)
    model("https://ultralytics.com/images/bus.jpg")
    model.export(format="onnx")

if __name__ == "__main__":
    main()
