"""Create a tiny synthetic ImageFolder dataset to smoke-test the pipeline without downloads."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

root = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
classes = ["Negative", "Broadleaf weed", "Grass weed"]
rng = np.random.default_rng(0)
for ci, c in enumerate(classes):
    os.makedirs(os.path.join(root, c), exist_ok=True)
    for i in range(40):
        img = Image.fromarray((rng.random((128, 128, 3)) * 60 + [90, 70, 40]).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for _ in range(ci * 6):
            x, y = rng.integers(0, 128, 2)
            if ci == 1:
                d.ellipse([x, y, x + 18, y + 12], fill=(30, 160, 40))
            else:
                d.line([x, y, x + 4, y - 30], fill=(60, 200, 60), width=3)
        img.save(os.path.join(root, c, f"{i:03d}.png"))
print(f"Synthetic dataset written to {root}")
