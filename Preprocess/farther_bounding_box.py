import os

folder = r"C:\junha\Datasets\LTDv2\Train_Labels"

top_y = None
top_file = None

for fname in os.listdir(folder):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(folder, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if len(t) < 5:
                continue
            y = float(t[2])
            if top_y is None or y < top_y:
                top_y = y
                top_file = fname

if top_file is not None:
    image_id = os.path.splitext(top_file)[0]
    print(f"image: {image_id}, top_y: {top_y}")
else:
    print("no boxes found")