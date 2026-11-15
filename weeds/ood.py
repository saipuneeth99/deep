"""Out-of-distribution (OOD) rejection: "is this even a field/plant image?"

A classifier trained on 9 classes must always answer one of those 9 - softmax spreads 100%
over them even for a photo of people. We reject such inputs with deep k-nearest-neighbours
(Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors", ICML 2022):

  1. take the CNN's penultimate feature vector (input of the final linear layer), L2-normalise it
  2. score = cosine distance to its k-th nearest neighbour among features of real training images
  3. reject if score > threshold, where threshold = 99th percentile of scores on validation images
     (so ~99% of genuine field images are accepted; far-away inputs like faces/rooms are rejected)
"""
import copy
import numpy as np
import torch.nn as nn


class WithFeatures(nn.Module):
    """Wraps a classifier so forward() returns (logits, penultimate features)."""

    def __init__(self, model):
        super().__init__()
        name = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]  # final layer
        self.backbone = copy.deepcopy(model)
        *parent, last = name.split(".")
        mod = self.backbone
        for p in parent:
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        self.fc = mod[int(last)] if last.isdigit() else getattr(mod, last)
        if last.isdigit():
            mod[int(last)] = nn.Identity()
        else:
            setattr(mod, last, nn.Identity())

    def forward(self, x):
        f = self.backbone(x)
        return self.fc(f), f


def normalise(f):
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def knn_distance(feats, bank, k):
    """Cosine distance from each feature to its k-th nearest neighbour in the (normalised) bank."""
    with np.errstate(all="ignore"):  # numpy 2.0 + macOS Accelerate emits spurious matmul warnings
        sims = normalise(feats.astype(np.float32)) @ bank.astype(np.float32).T
    kth = -np.partition(-sims, k - 1, axis=1)[:, k - 1]
    return 1.0 - kth
