from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO(r"/YOLO\runs\detect\train26\weights\best.pt")
results = model.predict(source=r"C:\junha\Git\RTIOD\Visualize\datasets\coco8\images\val")

for r in results:
    im = r.plot()[:, :, ::-1]  # BGR→RGB
    plt.figure(figsize=(8, 8))
    plt.imshow(im)
    plt.axis('off')
    plt.show()
