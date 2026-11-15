"""Dataset loading: DeepWeeds (labels.csv) or any ImageFolder-style dataset."""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEEPWEEDS_CLASSES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
                     "Rubber vine", "Siam weed", "Snake weed", "Negative"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OUTLIER_CLASS = "Not a field image"  # 10th class trained on people/rooms/objects (outlier exposure)
DEFAULT_OUTLIER_DIRS = ("data/outliers/images", "data/outliers_custom")


def get_transforms(img_size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(0.4, 0.4, 0.4, 0.08),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),  # camera motion / focus blur
            T.RandomGrayscale(0.05),
            T.ToTensor(), T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.2)])
    return T.Compose([T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def scan_dataset(root):
    """Returns (paths, labels, class_names). Supports DeepWeeds layout or ImageFolder."""
    csv = os.path.join(root, "labels.csv")
    if os.path.exists(csv):  # DeepWeeds: root/images/*.jpg + root/labels.csv
        df = pd.read_csv(csv)
        img_dir = os.path.join(root, "images") if os.path.isdir(os.path.join(root, "images")) else root
        paths = [os.path.join(img_dir, f) for f in df["Filename"]]
        return paths, df["Label"].astype(int).tolist(), DEEPWEEDS_CLASSES
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(root, c))):
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(root, c, f)); labels.append(i)
    if not paths:
        raise RuntimeError(f"No images found in {root}")
    return paths, labels, classes


class WeedDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def _split(n, labels, val, test, seed):
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test, stratify=labels, random_state=seed)
    tr, va = train_test_split(tr, test_size=val / (1 - test),
                              stratify=None if labels is None else [labels[i] for i in tr], random_state=seed)
    return tr, va, te


def scan_outliers(dirs):
    return sorted(os.path.join(d, f) for d in dirs if os.path.isdir(d)
                  for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def make_splits(root, val=0.15, test=0.15, seed=42, outlier_dirs=()):
    """70/15/15 stratified split. Outlier images (if any) are split separately with the same ratios
    and appended as an extra class, so the DeepWeeds split is identical with or without them."""
    paths, labels, classes = scan_dataset(root)
    splits = [([paths[i] for i in ids], [labels[i] for i in ids]) for ids in _split(len(paths), labels, val, test, seed)]
    out = scan_outliers(outlier_dirs)
    if out:
        k = len(classes)
        classes = list(classes) + [OUTLIER_CLASS]
        for (sp, sl), ids in zip(splits, _split(len(out), None, val, test, seed)):
            sp += [out[i] for i in ids]; sl += [k] * len(ids)
    return splits[0], splits[1], splits[2], classes


def splits_for(ckpt, root):
    """Re-create the exact splits a checkpoint was trained with."""
    return make_splits(root, outlier_dirs=ckpt.get("outlier_dirs", ()))


def make_loaders(root, img_size=224, batch_size=32, workers=4, balanced=True, seed=42, outlier_dirs=()):
    (trp, trl), (vap, val), (tep, tel), classes = make_splits(root, seed=seed, outlier_dirs=outlier_dirs)
    train_ds = WeedDataset(trp, trl, get_transforms(img_size, True))
    sampler = None
    if balanced:  # DeepWeeds is heavily imbalanced (Negative ~50%)
        counts = np.bincount(trl, minlength=len(classes)).astype(float)
        weights = (1.0 / np.maximum(counts, 1))[trl]
        sampler = WeightedRandomSampler(torch.as_tensor(weights), len(trl), replacement=True)
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=workers > 0)
    train = DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, drop_last=True, **kw)
    valid = DataLoader(WeedDataset(vap, val, get_transforms(img_size, False)), **kw)
    test = DataLoader(WeedDataset(tep, tel, get_transforms(img_size, False)), **kw)
    return train, valid, test, classes
