"""Build the out-of-distribution detector for a trained model (see weeds/ood.py).

    python calibrate_ood.py --run runs/mobilenet_v3_small

Saves runs/<arch>/ood.npz = {bank: normalised features of training images, k, threshold}.
Threshold = 99th percentile of kNN distance on validation images (99% of real images accepted).
(v1 used 97%; with the trained "Not a field image" class doing most of the rejection, 99% keeps more real weeds.)
The validation distances are stored too, so the app can change the acceptance rate live.
"""
import argparse, json, os
import cv2
import numpy as np
from realtime import Predictor
import torch
from weeds.data import OUTLIER_CLASS, splits_for
from weeds.ood import knn_distance, normalise


def features(pred, paths, bs=64):
    out = []
    for i in range(0, len(paths), bs):
        out.append(pred.infer([cv2.imread(p) for p in paths[i:i + bs]])[1])
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="data/deepweeds")
    p.add_argument("--bank", type=int, default=4000, help="training images kept in the feature bank")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--tpr", type=float, default=0.99, help="fraction of real images to accept")
    a = p.parse_args()

    pred = Predictor(os.path.join(a.run, "model.onnx"))  # FP32 features (same as PyTorch)
    pred.ood = None
    ckpt = torch.load(os.path.join(a.run, "best.pt"), map_location="cpu", weights_only=False)
    (trp, trl), (vap, val), _, classes = splits_for(ckpt, a.data)
    if OUTLIER_CLASS in classes:  # the feature bank must contain real field images only
        o = classes.index(OUTLIER_CLASS)
        trp = [p for p, l in zip(trp, trl) if l != o]; vap = [p for p, l in zip(vap, val) if l != o]
    rng = np.random.default_rng(0)
    bank_paths = [trp[i] for i in rng.choice(len(trp), min(a.bank, len(trp)), replace=False)]
    bank = normalise(features(pred, bank_paths)).astype(np.float16)
    val_d = knn_distance(features(pred, vap), bank, a.k)
    thr = float(np.quantile(val_d, a.tpr))
    np.savez_compressed(os.path.join(a.run, "ood.npz"), bank=bank, k=a.k, threshold=thr, val_dist=val_d)
    info = dict(bank=len(bank), dim=bank.shape[1], k=a.k, threshold=thr, accept_rate_val=float((val_d <= thr).mean()),
                val_dist_median=float(np.median(val_d)))
    json.dump(info, open(os.path.join(a.run, "ood_info.json"), "w"), indent=2)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
