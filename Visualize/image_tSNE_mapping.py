import os, re, glob, cv2, torch, numpy as np, pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

root = r"C:\junha\Datasets\LTDv2\frames\frames"
weights = r"C:\junha\Git\RTIOD\YOLO\runs\detect\train4\weights\best.pt"
meta_csv = r"C:\junha\Datasets\LTDv2\metadata_images.csv"
out_dir = r"C:\junha\Git\RTIOD\tSNE_Visualize2"

imgsz = 640
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
os.makedirs(out_dir, exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(42); np.random.seed(42)
if use_cuda: torch.cuda.manual_seed_all(42)

meta_cols_full = ["Folder name","Clip Name","Image Number","DateTime","Temperature","Humidity","Precipitation","Dew Point","Wind Direction","Wind Speed","Sun Radiation Intensity","Min of sunshine latest 10 min"]
meta = pd.read_csv(meta_csv, sep=None, engine="python")
for c in meta_cols_full:
    if c not in meta.columns: raise RuntimeError(f"missing meta col: {c}")

def build_path(row):
    date = str(row["Folder name"]).strip()
    clip = str(row["Clip Name"]).strip()
    imgid = str(row["Image Number"]).strip()
    base = imgid.replace(".jpg","").replace(".jpeg","")
    cands = [
        os.path.join(root, date, clip, base + ".jpg"),
        os.path.join(root, date, clip, base + ".jpeg"),
        os.path.join(root, date, clip, base + ".JPG"),
        os.path.join(root, date, clip, base + ".JPEG"),
        os.path.join(root, date, clip, base.replace("image_","img_") + ".jpg"),
        os.path.join(root, date, clip, base.replace("img_","image_") + ".jpg"),
    ]
    for p in cands:
        if os.path.exists(p): return os.path.normpath(p)
    g = glob.glob(os.path.join(root, date, clip, f"{base}*"))
    return os.path.normpath(g[0]) if g else np.nan

meta["path"] = meta.apply(build_path, axis=1)
meta = meta.dropna(subset=["path"])
meta["path"] = meta["path"].astype(str)
num_cols = ["Temperature","Humidity","Precipitation","Dew Point","Wind Direction","Wind Speed","Sun Radiation Intensity","Min of sunshine latest 10 min"]
for c in num_cols: meta[c] = pd.to_numeric(meta[c], errors="coerce")

model = YOLO(weights)
model.to(str(device))
backbone = model.model.eval()
layers_to_hook = [i for i in [16,19,22] if i < len(backbone.model)]
feat_buffers = []

def hook_fn(m, i, o):
    feat_buffers.append(o[0].detach() if isinstance(o, (list, tuple)) else o.detach())

hooks = [backbone.model[idx].register_forward_hook(hook_fn) for idx in layers_to_hook]

def preprocess(p):
    im = cv2.imread(p);
    if im is None: raise RuntimeError(f"read fail: {p}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    im = (im.astype(np.float32) / 255.0).transpose(2,0,1)
    return torch.from_numpy(im)[None].to(device, non_blocking=True)

def extract_feature(p):
    feat_buffers.clear()
    x = preprocess(p)
    with torch.inference_mode(): _ = backbone(x)
    feats = []
    for f in feat_buffers:
        f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        feats.append(f)
    return torch.cat(feats, dim=1).squeeze(0).cpu().numpy()

def load_thumb(p, size=64):
    im = Image.open(p).convert("RGB")
    im = im.resize((size, size))
    return np.asarray(im)

def safe_perplexity(n):
    base = min(30, max(5, n//3))
    return min(base, n-1)

def split_subclusters(df, min_samples=20):
    out = np.full(len(df), -1, dtype=int)
    for c, sub in df.groupby("cluster"):
        X = sub[["tsne_x","tsne_y"]].values
        if len(X) < max(10, min_samples*2):
            out[sub.index] = 0; continue
        k = min(8, len(X)-1)
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        d, _ = nn.kneighbors(X)
        eps = float(np.median(d[:, -1]) * 1.2)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        if labels.max() < 0: labels = np.zeros_like(labels)
        out[sub.index] = labels
    return out

def pick_reps(df):
    reps = []
    cen = df.groupby(["cluster","subcluster"])[["tsne_x","tsne_y"]].mean().reset_index()
    for _, row in cen.iterrows():
        c, s = int(row["cluster"]), int(row["subcluster"])
        sub = df[(df["cluster"]==c)&(df["subcluster"]==s)]
        d = (sub["tsne_x"]-row["tsne_x"])**2 + (sub["tsne_y"]-row["tsne_y"])**2
        p = sub.iloc[d.values.argmin()]["path"]
        reps.append((c, s, row["tsne_x"], row["tsne_y"], p))
    return reps

def meta_short(r):
    vals = []
    if pd.notna(r.get("Temperature", np.nan)): vals.append(f"T:{r['Temperature']:.1f}")
    if pd.notna(r.get("Humidity", np.nan)): vals.append(f"H:{r['Humidity']:.0f}")
    if pd.notna(r.get("Sun Radiation Intensity", np.nan)): vals.append(f"Sun:{r['Sun Radiation Intensity']:.0f}")
    return "  ".join(vals)

date_dirs = [d for d in sorted(os.listdir(root)) if re.fullmatch(r"\d{8}", d) and os.path.isdir(os.path.join(root, d))]
all_rows = []

for date in date_dirs:
    date_path = os.path.join(root, date)
    img_paths = []
    for clip in sorted(os.listdir(date_path)):
        clip_dir = os.path.join(date_path, clip)
        if os.path.isdir(clip_dir):
            img_paths.extend(sorted(glob.glob(os.path.join(clip_dir, "*.jpg"))))
    if not img_paths: continue
    X, kept = [], []
    for p in img_paths:
        try:
            X.append(extract_feature(p)); kept.append(p)
        except Exception: continue
    if len(X) < 5: continue
    X = np.stack(X)
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(2, min(10, len(X))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2: continue
        s = silhouette_score(X, labels)
        if s > best_score: best_k, best_score, best_labels = k, s, labels
    if best_labels is None: continue
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=safe_perplexity(len(X)))
    Z = tsne.fit_transform(X)
    df = pd.DataFrame({"date": date, "path": kept, "cluster": best_labels, "tsne_x": Z[:,0], "tsne_y": Z[:,1]})
    df = df.merge(meta[["path"] + meta_cols_full], on="path", how="left")
    df["subcluster"] = split_subclusters(df, min_samples=20)
    all_rows.append(df)

    num_cols = ["Temperature", "Humidity", "Precipitation", "Dew Point", "Wind Direction", "Wind Speed",
                "Sun Radiation Intensity", "Min of sunshine latest 10 min"]

    agg = {"path": "count"}
    for c in num_cols:
        if c in df.columns:
            agg[c] = "mean"

    df.groupby("cluster").agg(agg).rename(columns={"path": "count"}).reset_index().to_csv(os.path.join(out_dir, f"cluster_stats_{date}.csv"), index=False)

    df.groupby(["cluster", "subcluster"]).agg(agg).rename(columns={"path": "count"}).reset_index().to_csv(os.path.join(out_dir, f"subcluster_stats_{date}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12,7))
    ax.scatter(df["tsne_x"], df["tsne_y"], c=df["cluster"], s=10)
    ax.set_title(f"{date}  k={best_k}  silhouette={best_score:.3f}")
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")
    cent = df.groupby("cluster")[["tsne_x","tsne_y"]].mean()
    reps = []
    for k, rowc in cent.iterrows():
        sub = df[df["cluster"]==k]
        d = (sub["tsne_x"]-rowc["tsne_x"])**2 + (sub["tsne_y"]-rowc["tsne_y"])**2
        p = sub.iloc[d.values.argmin()]["path"]
        reps.append((k, rowc["tsne_x"], rowc["tsne_y"], p))
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
        except Exception:
            continue
        oi = OffsetImage(thumb, zoom=1)
        ab = AnnotationBbox(oi, (ix, cy), frameon=True, box_alignment=(0.5,0.5), pad=0.2)
        ax.add_artist(ab)
        ax.plot([cx, ix], [cy, cy], linewidth=0.8)
        r = df[df["path"]==p].iloc[0]
        ax.text(ix, cy-0.08*yspan, f"{os.path.basename(p)}\n{meta_short(r)}", ha="center", va="top", fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"tsne_{date}.png"), dpi=200); plt.close()

    fig2, ax2 = plt.subplots(figsize=(12,7))
    ax2.scatter(df["tsne_x"], df["tsne_y"], c=df["subcluster"], s=10, cmap="tab20")
    ax2.set_title(f"{date}  subclusters")
    ax2.set_xlabel("tSNE-1"); ax2.set_ylabel("tSNE-2")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"tsne_sub_{date}.png"), dpi=200); plt.close()

    meta_key = "Temperature"
    if meta_key in df.columns:
        fig3, ax3 = plt.subplots(figsize=(12,7))
        sc = ax3.scatter(df["tsne_x"], df["tsne_y"], c=df[meta_key], s=10)
        ax3.set_title(f"{date}  color={meta_key}")
        ax3.set_xlabel("tSNE-1"); ax3.set_ylabel("tSNE-2")
        cbar = plt.colorbar(sc, ax=ax3); cbar.set_label(meta_key)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"tsne_meta_{meta_key}_{date}.png"), dpi=200); plt.close()

    for (c, s), sub in df.groupby(["cluster","subcluster"]):
        sub.to_csv(os.path.join(out_dir, f"subcluster_{date}_C{c}_S{s}.csv"), index=False)
    df.to_csv(os.path.join(out_dir, f"features_{date}.csv"), index=False)

if all_rows:
    pd.concat(all_rows, ignore_index=True).to_csv(os.path.join(out_dir, "features_all_dates.csv"), index=False)

for h in hooks: h.remove()
