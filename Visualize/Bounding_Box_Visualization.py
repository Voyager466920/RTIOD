from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

img_path = Path(r"C:\junha\Datasets\LTDv2\frames\frames\20200611\clip_37_1620\image_0020.jpg")
label_path = Path(r"C:\junha\Datasets\LTDv2\Train_Labels\20200603711162020.txt")

im = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(im)
W, H = im.size

with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        t = line.strip().split()
        if len(t) < 5:
            continue
        c, x, y, w, h = map(float, t[:5])
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x, y - 10), str(int(c)), fill=(255, 255, 255))

out_path = img_path.with_name(img_path.stem + "_viz.jpg")
im.save(out_path)
print("saved:", out_path)
