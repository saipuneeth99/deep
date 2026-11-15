"""Download DeepWeeds (17,509 images, 8 weed species + Negative, 256x256) and convert it to
    data/deepweeds/images/*.jpg + data/deepweeds/labels.csv   (Filename, Label, Species)

The original host (nextcloud.qriscloud.org.au) is offline, so this uses the Hugging Face
mirror published by Project-AgML (CC-BY-4.0).
Olsen et al., "DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning", Sci. Rep. 2019.
"""
import io, os, sys, urllib.request
import pandas as pd
import pyarrow.parquet as pq

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/deepweeds"
URL = ("https://huggingface.co/datasets/Project-AgML/DeepWeeds_classification/resolve/main/"
       "data/train-00000-of-00001.parquet")
PARQUET = os.path.join("data", "raw_download", "deepweeds.parquet")


def progress(n, bs, total):
    if total > 0:
        print(f"\r  {min(n*bs, total)/1e6:7.1f} / {total/1e6:.1f} MB", end="")


if not os.path.exists(PARQUET):
    os.makedirs(os.path.dirname(PARQUET), exist_ok=True)
    print("Downloading DeepWeeds parquet (~490 MB) ...")
    urllib.request.urlretrieve(URL, PARQUET, progress); print()

img_dir = os.path.join(ROOT, "images")
os.makedirs(img_dir, exist_ok=True)
pf = pq.ParquetFile(PARQUET)
rows = []
for batch in pf.iter_batches(batch_size=500, columns=["image", "label", "species"]):
    for rec in batch.to_pylist():
        name = f"{len(rows):05d}.jpg"
        img = rec["image"]
        data = img["bytes"] if isinstance(img, dict) else img
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(data)
        rows.append((name, int(rec["label"]), rec["species"]))
    print(f"\r  extracted {len(rows)} images", end="")
print()
df = pd.DataFrame(rows, columns=["Filename", "Label", "Species"])
df.to_csv(os.path.join(ROOT, "labels.csv"), index=False)
print(df.groupby(["Label", "Species"]).size().to_string())
print(f"Done -> {ROOT}")
