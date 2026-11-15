"""Make a demo 'drone/tractor pass' video: a camera window panning over a large field mosaic
built from unseen test-set images (~50% weeds), for testing realtime.py and the Streamlit video tab.

    python scripts/make_demo_video.py data/demo_field.mp4
"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weeds.data import make_splits

out = sys.argv[1] if len(sys.argv) > 1 else "data/demo_field.mp4"
_, _, (paths, labels), classes = make_splits("data/deepweeds")
rng = np.random.default_rng(7)
neg = [i for i, l in enumerate(labels) if classes[l] == "Negative"]
pos = [i for i, l in enumerate(labels) if classes[l] != "Negative"]
R, C, T = 3, 16, 240  # field = 3 rows x 16 cols of 240px tiles
tiles = [cv2.resize(cv2.imread(paths[int(rng.choice(neg if rng.random() < 0.5 else pos))]), (T, T))
         for _ in range(R * C)]
field = np.vstack([np.hstack(tiles[r * C:(r + 1) * C]) for r in range(R)])
W = 4 * T  # camera sees 3 x 4 tiles
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, R * T))
for x in range(0, field.shape[1] - W, 8):
    writer.write(field[:, x:x + W])
writer.release()
print(f"demo video -> {out}")
