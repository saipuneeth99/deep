"""Train a lightweight CNN for weed classification.

Example:
    python train.py --data data/deepweeds --arch mobilenet_v3_small --epochs 30
"""
import argparse, json, os, time
import torch
import torch.nn as nn
from tqdm import tqdm

from torchvision.transforms import v2
from weeds.data import DEFAULT_OUTLIER_DIRS, make_loaders, scan_outliers
from weeds.models import ARCHS, build_model, count_params
from weeds.utils import get_device


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, mix=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "eval"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            target = y
            if train and mix is not None and torch.rand(1).item() < 0.5:
                x, target = mix(x, y)  # MixUp or CutMix -> soft targets
            out = model(x)
            loss = criterion(out, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/deepweeds")
    p.add_argument("--arch", default="mobilenet_v3_small", choices=ARCHS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--outliers", nargs="*", default=list(DEFAULT_OUTLIER_DIRS),
                   help="folders of non-field images -> extra class 'Not a field image' (pass nothing to disable)")
    p.add_argument("--no-mix", action="store_true", help="disable MixUp/CutMix")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    torch.manual_seed(42)
    device = get_device(a.device)
    outlier_dirs = [d for d in a.outliers if os.path.isdir(d)]
    train_dl, val_dl, _, classes = make_loaders(a.data, a.img_size, a.batch_size, a.workers,
                                                outlier_dirs=outlier_dirs)
    mix = None if a.no_mix else v2.RandomChoice([v2.MixUp(num_classes=len(classes)),
                                                  v2.CutMix(num_classes=len(classes))])
    print(f"outlier images: {len(scan_outliers(outlier_dirs))} from {outlier_dirs}  mixup/cutmix: {mix is not None}")
    model = build_model(a.arch, len(classes), pretrained=not a.no_pretrained).to(device)
    print(f"Device={device}  arch={a.arch}  params={count_params(model)/1e6:.2f}M  classes={len(classes)}")
    print(f"train={len(train_dl.dataset)} val={len(val_dl.dataset)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=a.lr, total_steps=a.epochs * len(train_dl), pct_start=0.1)

    out_dir = os.path.join(a.out, a.arch)
    os.makedirs(out_dir, exist_ok=True)
    best, bad, history = 0.0, 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer, scheduler, mix)
        vl, va = run_epoch(model, val_dl, criterion, device)
        history.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        flag = ""
        if va > best:
            best, bad, flag = va, 0, " *"
            torch.save(dict(model=model.state_dict(), arch=a.arch, classes=classes,
                            img_size=a.img_size, val_acc=va, epoch=ep, outlier_dirs=outlier_dirs), os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
        print(f"[{ep:3d}/{a.epochs}] train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}  {time.time()-t0:.0f}s{flag}")
        if bad >= a.patience:
            print("Early stopping."); break
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    _plot(history, out_dir)
    print(f"Best val acc {best:.4f}. Checkpoint: {out_dir}/best.pt")


def _plot(h, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, axi in (("loss", ax[0]), ("acc", ax[1])):
        axi.plot(e, [r[f"train_{k}"] for r in h], label="train")
        axi.plot(e, [r[f"val_{k}"] for r in h], label="val")
        axi.set_title(k); axi.set_xlabel("epoch"); axi.legend(); axi.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "curves.png"), dpi=120)


if __name__ == "__main__":
    main()
