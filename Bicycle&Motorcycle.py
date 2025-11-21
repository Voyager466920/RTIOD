import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

json_path = Path(r'C:\junha\Datasets\LTDv2\Train.json')

with open(json_path, encoding='utf-8') as f:
    data = json.load(f)

images = {img['id']: img['file_name'] for img in data['images']}

bike_ids = set()
moto_ids = set()

for ann in data['annotations']:
    if ann['category_id'] == 2:
        bike_ids.add(ann['image_id'])
    elif ann['category_id'] == 3:
        moto_ids.add(ann['image_id'])

month_bike = {}
month_moto = {}

for img_id in bike_ids:
    f = images.get(img_id)
    if not f:
        continue
    date = f.split('/')[1]
    month = date[:6]
    month_bike[month] = month_bike.get(month, 0) + 1

for img_id in moto_ids:
    f = images.get(img_id)
    if not f:
        continue
    date = f.split('/')[1]
    month = date[:6]
    month_moto[month] = month_moto.get(month, 0) + 1

months = sorted(set(list(month_bike.keys()) + list(month_moto.keys())))
bike_counts = [month_bike.get(m, 0) for m in months]
moto_counts = [month_moto.get(m, 0) for m in months]

x = np.arange(len(months))
w = 0.4

plt.figure(figsize=(14,6))
plt.bar(x - w/2, bike_counts)
plt.bar(x + w/2, moto_counts)
for i,c in enumerate(bike_counts):
    plt.text(x[i] - w/2, c, str(c), ha='center', va='bottom', fontsize=8)
for i,c in enumerate(moto_counts):
    plt.text(x[i] + w/2, c, str(c), ha='center', va='bottom', fontsize=8)
plt.xticks(x, months, rotation=45)
plt.xlabel('month')
plt.ylabel('count')
plt.tight_layout()
plt.show()
