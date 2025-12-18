import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

CITYSCAPES_ROOT = "/Users/daniil_deltsov/Documents/university/HSE/Thesis/cityscapes"
SPLITS = ["train", "val"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = [
    ("road",          (128, 64, 128)),
    ("sidewalk",      (244, 35, 232)),
    ("building",      (70,  70,  70)),
    ("wall",          (102, 102, 156)),
    ("fence",         (190, 153, 153)),
    ("pole",          (153, 153, 153)),
    ("traffic light", (250, 170,  30)),
    ("traffic sign",  (220, 220,   0)),
    ("vegetation",    (107, 142,  35)),
    ("terrain",       (152, 251, 152)),
    ("sky",           (70,  130, 180)),
    ("person",        (220,  20,  60)),
    ("rider",         (255,   0,   0)),
    ("car",           (0,    0,  142)),
    ("truck",         (0,    0,   70)),
    ("bus",           (0,   60, 100)),
    ("train",         (0,   80, 100)),
    ("motorcycle",    (0,    0,  230)),
    ("bicycle",       (119,  11,  32)),
]

TRAINID_TO_NAME = {i: cls[0] for i, cls in enumerate(CLASSES)}
TRAINID_TO_COLOR = {i: cls[1] for i, cls in enumerate(CLASSES)}
CLASS_NAMES = [c[0] for c in CLASSES]
N_CLASSES = len(CLASSES)

# _gtFine_labelIds.png always exists; _gtFine_labelTrainIds.png requires a separate prep script
LABELID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

def labelids_to_trainids(lbl_array):
    out = np.full_like(lbl_array, 255, dtype=np.uint8)
    for lid, tid in LABELID_TO_TRAINID.items():
        out[lbl_array == lid] = tid
    return out

def label_to_color(lbl_array):
    h, w = lbl_array.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in TRAINID_TO_COLOR.items():
        color_map[lbl_array == cid] = color
    return color_map


split_stats = {}
all_cities = set()
image_paths = []
label_paths = []

for split in SPLITS:
    img_root = os.path.join(CITYSCAPES_ROOT, "leftImg8bit", split)
    lbl_root = os.path.join(CITYSCAPES_ROOT, "gtFine", split)

    cities = sorted([
        d for d in os.listdir(img_root)
        if not d.startswith(".") and os.path.isdir(os.path.join(img_root, d))
    ]) if os.path.exists(img_root) else []
    all_cities.update(cities)

    imgs, lbls = [], []
    for city in cities:
        city_imgs = sorted([
            os.path.join(img_root, city, f)
            for f in os.listdir(os.path.join(img_root, city))
            if f.endswith("_leftImg8bit.png") and not f.startswith(".")
        ])
        city_lbls = [
            os.path.join(lbl_root, city,
                         os.path.basename(p).replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
            for p in city_imgs
        ]
        imgs.extend(city_imgs)
        lbls.extend(city_lbls)

    split_stats[split] = {"n_images": len(imgs), "cities": cities}
    if split == "train":
        image_paths = imgs
        label_paths = lbls

    print(f"{split}: {len(imgs)} images, {len(cities)} cities")


sample_n = min(50, len(image_paths))
sample_imgs = np.random.choice(image_paths, sample_n, replace=False)

heights, widths, channel_means, channel_stds = [], [], [], []

for p in tqdm(sample_imgs, desc="image stats"):
    img = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
    heights.append(img.shape[0])
    widths.append(img.shape[1])
    channel_means.append(img.mean(axis=(0, 1)))
    channel_stds.append(img.std(axis=(0, 1)))

mean_rgb = np.array(channel_means).mean(axis=0) / 255.0
std_rgb  = np.array(channel_stds).mean(axis=0) / 255.0

print(f"resolution: {int(np.mean(heights))}x{int(np.mean(widths))}")
print(f"mean RGB: {mean_rgb.round(4)}")
print(f"std  RGB: {std_rgb.round(4)}")


sample_n_cls = min(200, len(label_paths))
sample_lbls = np.random.choice(label_paths, sample_n_cls, replace=False)

pixel_counts = np.zeros(N_CLASSES, dtype=np.int64)
total_pixels = 0
found_labels = 0

for p in tqdm(sample_lbls, desc="pixel counts"):
    if not os.path.exists(p):
        continue
    lbl = labelids_to_trainids(np.array(Image.open(p)))
    for cid in range(N_CLASSES):
        pixel_counts[cid] += (lbl == cid).sum()
    total_pixels += lbl.size
    found_labels += 1

if found_labels == 0:
    print("no label files found — check CITYSCAPES_ROOT")
else:
    print(f"loaded {found_labels} label files")

pixel_pct = (pixel_counts / total_pixels * 100) if total_pixels > 0 else np.zeros(N_CLASSES)

print(f"\n{'class':<18} {'pixels':>12}  {'%':>6}")
print("-" * 40)
for i, name in enumerate(CLASS_NAMES):
    pct = pixel_pct[i] if not np.isnan(pixel_pct[i]) else 0.0
    bar = "█" * int(pct / 2)
    print(f"{name:<18} {pixel_counts[i]:>12,}  {pct:>5.2f}%  {bar}")

fig, ax = plt.subplots(figsize=(12, 6))
colors = [np.array(TRAINID_TO_COLOR[i]) / 255.0 for i in range(N_CLASSES)]
ax.barh(CLASS_NAMES[::-1], pixel_pct[::-1], color=colors[::-1])
ax.set_xlabel("Pixel percentage (%)")
ax.set_title("Cityscapes — Pixel-level Class Distribution (train sample)")
ax.axvline(100 / N_CLASSES, color="red", linestyle="--", alpha=0.5, label="uniform baseline")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)


city_counts = defaultdict(int)
for p in image_paths:
    city = os.path.basename(p).split("_")[0]
    city_counts[city] += 1

cities_sorted = sorted(city_counts.items(), key=lambda x: -x[1])
for city, cnt in cities_sorted:
    print(f"{city:<20} {cnt:3d}")

fig, ax = plt.subplots(figsize=(10, 5))
city_names, city_vals = zip(*cities_sorted)
ax.bar(city_names, city_vals, color=sns.color_palette("muted", len(city_names)))
ax.set_xlabel("City")
ax.set_ylabel("Images")
ax.set_title("Cityscapes — Images per City (train)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "city_distribution.png"), dpi=150)


n_samples = min(6, len(image_paths))
sample_indices = np.random.choice(len(image_paths), n_samples, replace=False)

fig, axes = plt.subplots(n_samples, 2, figsize=(14, n_samples * 3))
fig.suptitle("Cityscapes — Sample Images & Segmentation Masks")

for i, idx in enumerate(sample_indices):
    img = Image.open(image_paths[idx]).convert("RGB")
    lbl_path = label_paths[idx]

    axes[i, 0].imshow(img)
    axes[i, 0].set_title(os.path.basename(image_paths[idx])[:35], fontsize=8)
    axes[i, 0].axis("off")

    if os.path.exists(lbl_path):
        lbl = labelids_to_trainids(np.array(Image.open(lbl_path)))
        axes[i, 1].imshow(label_to_color(lbl))
        axes[i, 1].set_title("segmentation mask", fontsize=8)
    else:
        axes[i, 1].text(0.5, 0.5, "label not found", ha="center", va="center")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=120)

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.axis("off")
patches = [mpatches.Patch(color=np.array(c) / 255.0, label=n) for n, c in CLASSES]
ax2.legend(handles=patches, loc="center", ncol=4, fontsize=10,
           title="Cityscapes eval classes (19)", title_fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_legend.png"), dpi=120)


rare_threshold = 1.0
rare_classes   = [(CLASS_NAMES[i], pixel_pct[i]) for i in range(N_CLASSES) if pixel_pct[i] < rare_threshold]
common_classes = [(CLASS_NAMES[i], pixel_pct[i]) for i in range(N_CLASSES) if pixel_pct[i] >= rare_threshold]

print("\ncommon classes (>=1%):")
for name, pct in sorted(common_classes, key=lambda x: -x[1]):
    print(f"  {name:<20} {pct:.2f}%")

print("\nrare classes (<1%):")
for name, pct in sorted(rare_classes, key=lambda x: x[1]):
    print(f"  {name:<20} {pct:.3f}%")

print(f"\ndone — outputs in {OUTPUT_DIR}")
