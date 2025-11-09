import os, re, glob, cv2, torch, numpy as np, pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

root = r"C:\junha\Datasets\LTDv2\frames\frames"
weights = r"C:\junha\Git\RTIOD\YOLO\runs\detect\train4\weights\best.pt"

imgsz = 640
device = 0 if torch.cuda.is_available() else "cpu"
out_dir = r"/tSNE_Visualize\runs\features_tsne"
os.makedirs(out_dir, exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(42)
np.random.seed(42)

model = YOLO(weights)
model.to(device)
backbone = model.model
layers_to_hook = [16, 19, 22]
feat_buffers = []

def hook_fn(m, i, o):
    feat_buffers.append(o)

hooks = [backbone.model[idx].register_forward_hook(hook_fn) for idx in layers_to_hook]

def preprocess(p):
    im = cv2.imread(p)
    if im is None:
        raise RuntimeError(f"read fail: {p}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))
    return torch.from_numpy(im)[None].to(device)

def extract_feature(p):
    feat_buffers.clear()
    x = preprocess(p)
    _ = backbone(x)
    feats = []
    for f in feat_buffers:
        if isinstance(f, (list, tuple)):
            f = f[0]
        f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        feats.append(f)
    return torch.cat(feats, dim=1).squeeze(0).cpu().numpy()

def load_thumb(p, size=64):
    im = Image.open(p).convert("RGB")
    im = im.resize((size, size))
    return np.asarray(im)

date_dirs = [d for d in sorted(os.listdir(root)) if re.fullmatch(r"\d{8}", d) and os.path.isdir(os.path.join(root, d))]
all_rows = []

for date in date_dirs:
    date_path = os.path.join(root, date)
    img_paths = []
    for clip in sorted(os.listdir(date_path)):
        clip_dir = os.path.join(date_path, clip)
        if os.path.isdir(clip_dir):
            img_paths.extend(sorted(glob.glob(os.path.join(clip_dir, "*.jpg"))))
    if not img_paths:
        continue
    X, kept_paths = [], []
    for p in img_paths:
        try:
            X.append(extract_feature(p))
            kept_paths.append(p)
        except:
            continue
    if len(X) < 5:
        continue
    X = np.stack(X)
    best_k, best_score, best_labels = None, -1, None
    for k in range(2, min(10, len(X))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_score:
            best_k, best_score, best_labels = k, s, labels
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=max(5, min(30, len(X)//3)))
    Z = tsne.fit_transform(X)
    df = pd.DataFrame({"date": date, "path": kept_paths, "cluster": best_labels, "tsne_x": Z[:, 0], "tsne_y": Z[:, 1]})
    all_rows.append(df)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(df["tsne_x"], df["tsne_y"], c=df["cluster"], s=10)
    ax.set_title(f"{date}  k={best_k}  silhouette={best_score:.3f}")
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")

    cent = df.groupby("cluster")[["tsne_x","tsne_y"]].mean()
    reps = []
    for k, row in cent.iterrows():
        sub = df[df["cluster"]==k]
        d = (sub["tsne_x"]-row["tsne_x"])**2 + (sub["tsne_y"]-row["tsne_y"])**2
        idx = d.values.argmin()
        p = sub.iloc[idx]["path"]
        reps.append((k, row["tsne_x"], row["tsne_y"], p))
    reps.sort(key=lambda x: x[2])

    xr = df["tsne_x"].max() - df["tsne_x"].min()
    xl = df["tsne_x"].min() - 0.25*xr
    xr_ext = df["tsne_x"].max() + 0.25*xr
    ax.set_xlim(xl-0.05*xr, xr_ext+0.05*xr)

    yspan = df["tsne_y"].max() - df["tsne_y"].min()
    for idx, (k, cx, cy, p) in enumerate(reps):
        side = "left" if idx%2==0 else "right"
        ix = xl if side=="left" else xr_ext
        try:
            thumb = load_thumb(p, size=64)
        except:
            continue
        oi = OffsetImage(thumb, zoom=1)
        ab = AnnotationBbox(oi, (ix, cy), frameon=True, box_alignment=(0.5,0.5), pad=0.2)
        ax.add_artist(ab)
        ax.plot([cx, ix], [cy, cy], linewidth=0.8)
        ax.text(ix, cy-0.08*yspan, os.path.basename(p), ha="center", va="top", fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tsne_{date}.png"), dpi=200)
    plt.close()
    df.to_csv(os.path.join(out_dir, f"features_{date}.csv"), index=False)

if all_rows:
    pd.concat(all_rows, ignore_index=True).to_csv(os.path.join(out_dir, "features_all_dates.csv"), index=False)

for h in hooks:
    h.remove()
