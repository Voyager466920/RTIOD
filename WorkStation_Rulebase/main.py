import os
import glob
import cv2
import numpy as np

def get_diff_mask(prev, curr):
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(curr, prev)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, diff_bin = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return diff_bin

def visualize_pixel_diff(prev, curr):
    mask = get_diff_mask(prev, curr)
    vis = curr.copy()
    vis[mask > 0] = [0, 0, 255]
    return vis

def run_on_clip(clip_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(clip_dir, "image_*.jpg")))
    if len(files) < 2:
        return
    prev = cv2.imread(files[0])
    for idx in range(1, len(files)):
        curr_path = files[idx]
        curr = cv2.imread(curr_path)
        if prev is None or curr is None:
            prev = curr
            continue
        vis = visualize_pixel_diff(prev, curr)
        out_path = os.path.join(out_dir, os.path.basename(curr_path))
        cv2.imwrite(out_path, vis)
        prev = curr

if __name__ == "__main__":
    clip_dir = r"C:\junha\Datasets\LTDv2\frames\frames\20200514\clip_0_1331"
    out_dir = r"./20200514_clip_0_1331_pixel"
    run_on_clip(clip_dir, out_dir)
