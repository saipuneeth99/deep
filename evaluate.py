"""Evaluate a checkpoint on the held-out test split: accuracy, F1, confusion matrix."""
import argparse, json, os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from weeds.data import OUTLIER_CLASS, make_loaders
from weeds.utils import get_device, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/deepweeds")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    device = get_device(a.device)
    model, ckpt = load_checkpoint(a.ckpt, device)
    _, _, test_dl, classes = make_loaders(a.data, ckpt["img_size"], 64, a.workers,
                                          outlier_dirs=ckpt.get("outlier_dirs", ()))
    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_dl:
            ps.append(model(x.to(device)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    y, pr = np.concatenate(ys), np.concatenate(ps)
    labels = list(range(len(classes)))
    report = classification_report(y, pr, labels=labels, target_names=classes, digits=4, zero_division=0)
    print(report)
    res = dict(accuracy=accuracy_score(y, pr), macro_f1=f1_score(y, pr, average="macro"),
               weighted_f1=f1_score(y, pr, average="weighted"))
    if OUTLIER_CLASS in classes:  # report DeepWeeds-only numbers too, comparable with the 9-class models
        o = classes.index(OUTLIER_CLASS); dw = y != o
        res.update(deepweeds_acc=accuracy_score(y[dw], pr[dw]),
                   deepweeds_macro_f1=f1_score(y[dw], pr[dw], labels=list(range(o)), average="macro"),
                   field_images_rejected_as_outlier=float((pr[dw] == o).mean()),
                   outlier_recall=float((pr[~dw] == o).mean()))
    print(json.dumps(res, indent=2))

    out_dir = os.path.dirname(a.ckpt)
    json.dump(res, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
    open(os.path.join(out_dir, "test_report.txt"), "w").write(report)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pr, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(labels); ax.set_yticks(labels)
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix — {ckpt['arch']}")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=130)
    print(f"Saved metrics + confusion matrix to {out_dir}")


if __name__ == "__main__":
    main()
