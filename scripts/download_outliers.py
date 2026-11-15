"""Download a non-field "outlier" image set for training a 10th class, "Not a field image"
(outlier exposure; Hendrycks et al., ICLR 2019). No training happens here.

Sources (Hugging Face mirrors, downloaded with curl and TLS verification on):
  * COCO val2017 (one shard, ~2.5k images): people, faces, objects, streets, vehicles, food
  * MIT Indoor Scenes (Quattoni & Torralba, CVPR 2009): ~30 images from each of 65 indoor scene types
    (classroom, office, computer room, lab, corridor, library ...; greenhouse and florist are skipped)

Images where green vegetation covers > 15% of pixels (Excess-Green index) are dropped, so the
outlier class never learns that "plants = not a field image".

Output:
    data/outliers/images/{coco,indoor}_*.jpg   (longest side <= 512 px)
    data/outliers/manifest.csv                 (Filename, Source, Category, VegFraction)
    data/outliers_custom/                      (put your own classroom/lab/face photos here)

    python scripts/download_outliers.py
"""
import io, json, os, random, subprocess, sys, urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

OUT = "data/outliers"
RAW = "data/raw_download"
HF = "https://huggingface.co"
COCO_SHARD = "datasets/detection-datasets/coco/resolve/main/data/val-00001-of-00002-7af5414a3b178949.parquet"
MIT_REPO = "u5753411/MIT-Indoor-Scenes"
MIT_ROOT = "indoorCVPR_09/Images"
SKIP_SCENES = {"greenhouse", "florist"}
PER_SCENE, N_COCO, VEG_MAX, MAX_SIDE = 30, 2200, 0.15, 512


def curl(url, out=None):
    cmd = ["curl", "-sSfL", "--retry", "6", "--retry-all-errors", "--retry-delay", "2", "--max-time", "900", url]
    if out:
        subprocess.run(cmd + ["-o", out], check=True)
        return None
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def veg_fraction(img):
    """Fraction of pixels that are green vegetation: ExG = 2g - r - b on chromatic coordinates."""
    a = np.asarray(img.convert("RGB").resize((128, 128)), np.float32)
    s = a.sum(2, keepdims=True) + 1e-6
    r, g, b = np.moveaxis(a / s, 2, 0)
    return float(((2 * g - r - b) > 0.1).mean())


def save(img, name):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(os.path.join(OUT, "images", name), quality=90)


def coco(rows):
    import pyarrow.parquet as pq
    path = os.path.join(RAW, "coco_val_shard1.parquet")
    if not os.path.exists(path):
        print("Downloading COCO val2017 shard (~400 MB) ...")
        curl(f"{HF}/{COCO_SHARD}", path)
    kept = dropped = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200, columns=["image"]):
        for rec in batch.to_pylist():
            if kept >= N_COCO:
                return dropped
            im = rec["image"]
            img = Image.open(io.BytesIO(im["bytes"] if isinstance(im, dict) else im))
            v = veg_fraction(img)
            if v > VEG_MAX:
                dropped += 1
                continue
            name = f"coco_{kept:05d}.jpg"
            save(img, name)
            rows.append((name, "coco_val2017", "", round(v, 3)))
            kept += 1
    return dropped


def mit(rows):
    tree = json.loads(curl(f"{HF}/api/datasets/{MIT_REPO}/tree/main/{MIT_ROOT}"))
    scenes = sorted(os.path.basename(x["path"]) for x in tree if x["type"] == "directory")
    scenes = [s for s in scenes if s not in SKIP_SCENES]
    print(f"MIT Indoor: {len(scenes)} scene types")
    rng = random.Random(0)

    def scene_files(s):
        files = json.loads(curl(f"{HF}/api/datasets/{MIT_REPO}/tree/main/{MIT_ROOT}/{s}"))
        files = sorted(x["path"] for x in files if x["path"].lower().endswith((".jpg", ".jpeg", ".png")))
        return s, rng.sample(files, min(PER_SCENE, len(files)))

    with ThreadPoolExecutor(8) as ex:
        jobs = [(s, f) for s, fs in ex.map(scene_files, scenes) for f in fs]

    def fetch(job):
        s, f = job
        try:
            data = curl(f"{HF}/datasets/{MIT_REPO}/resolve/main/{urllib.parse.quote(f)}")
            img = Image.open(io.BytesIO(data))
            img.load()  # fully decode here so corrupt files are skipped
            return s, img.convert("RGB")
        except Exception:
            return s, None

    dropped = failed = 0
    with ThreadPoolExecutor(8) as ex:
        for i, (s, img) in enumerate(ex.map(fetch, jobs)):
            if img is None:
                failed += 1
                continue
            v = veg_fraction(img)
            if v > VEG_MAX:
                dropped += 1
                continue
            name = f"indoor_{s}_{i:05d}.jpg"
            save(img, name)
            rows.append((name, "mit_indoor", s, round(v, 3)))
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    return dropped, failed


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    os.makedirs("data/outliers_custom", exist_ok=True)
    with open("data/outliers_custom/README.txt", "w") as f:
        f.write("Put your own non-field photos here (classroom, lab, faces, desks, screens, hands...).\n"
                "Any .jpg/.png works. They will be added to the 'Not a field image' class at training time.\n"
                "Photos from the same webcam/phone you will demo with help the most.\n")
    rows = []
    d_coco = coco(rows)
    d_mit, failed = mit(rows)
    df = pd.DataFrame(rows, columns=["Filename", "Source", "Category", "VegFraction"])
    df.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print(df.groupby("Source").size().to_string())
    print(f"Total {len(df)} images. Dropped for vegetation: COCO {d_coco}, MIT {d_mit}. MIT download failures: {failed}")


if __name__ == "__main__":
    sys.exit(main())
