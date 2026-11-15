"""Real-time weed detection from webcam / video / image on edge devices.

The frame is split into an N x M grid of tiles; every tile is classified in one
batched forward pass by the lightweight CNN. Tiles predicted as a weed species
(with confidence >= threshold) are boxed and labelled, giving a coarse weed
localisation map suitable for targeted spraying, at real-time speed.

    python realtime.py --model runs/mobilenet_v3_small/model.int8.onnx --source 0
    python realtime.py --model runs/mobilenet_v3_small/best.pt --source field.mp4 --save out.mp4
    python realtime.py --model runs/weednet_lite/model.onnx --source photo.jpg --grid 4 4
"""
import argparse, os, time
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Predictor:
    """Runs a .pt or .onnx model. Returns class probabilities, and - when runs/<arch>/ood.npz exists -
    flags inputs that are out-of-distribution (not field/plant images) via deep kNN."""

    def __init__(self, path, threads=4):
        if path.endswith(".onnx"):
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.img_size = self.sess.get_inputs()[0].shape[2]
            cls_file = os.path.join(os.path.dirname(path), "classes.txt")
            self.classes = open(cls_file).read().splitlines()
            self.kind = "onnx"
        else:
            import torch
            from weeds.utils import load_checkpoint, get_device
            torch.set_num_threads(threads)
            from weeds.ood import WithFeatures
            self.device = get_device()
            model, ckpt = load_checkpoint(path, self.device)
            self.model = WithFeatures(model).to(self.device).eval()
            self.classes, self.img_size = ckpt["classes"], ckpt["img_size"]
            self.kind = "torch"
        ood_file = os.path.join(os.path.dirname(path), "ood.npz")
        self.ood = dict(np.load(ood_file)) if os.path.exists(ood_file) else None
        self.ood_threshold = float(self.ood["threshold"]) if self.ood else None
        self.colors = {i: tuple(int(c) for c in np.random.RandomState(i + 3).randint(60, 255, 3))
                       for i in range(len(self.classes))}

    def preprocess(self, tiles):
        b = np.stack([cv2.resize(t, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                      for t in tiles])
        b = (b[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD  # BGR->RGB, normalise
        return np.ascontiguousarray(b.transpose(0, 3, 1, 2))

    def infer(self, tiles):
        """Returns (probs [N,C], features [N,D] or None)."""
        x = self.preprocess(tiles)
        if self.kind == "onnx":
            out = self.sess.run(None, {self.inp: x})
            logits, feats = out[0], (out[1] if len(out) > 1 else None)
        else:
            import torch
            with torch.inference_mode():
                logits, feats = (t.float().cpu().numpy() for t in self.model(torch.from_numpy(x).to(self.device)))
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), feats

    def __call__(self, tiles):
        return self.infer(tiles)[0]

    def analyse(self, tiles):
        """Returns (probs, ood_distance or None, is_ood bool array)."""
        probs, feats = self.infer(tiles)
        if self.ood is None or feats is None:
            return probs, None, np.zeros(len(probs), bool)
        from weeds.ood import knn_distance
        d = knn_distance(feats, self.ood["bank"], int(self.ood["k"]))
        return probs, d, d > self.ood_threshold


def is_weed(name):
    return name.lower() not in ("negative", "background", "crop", "soil", "none", "not a field image")


def detect_tiles(frame, pred, rows, cols, with_ood=False):
    """Split frame into rows x cols tiles and classify all tiles in ONE batched forward pass."""
    h, w = frame.shape[:2]
    th, tw = h // rows, w // cols
    boxes = [(r * th, c * tw, (r + 1) * th, (c + 1) * tw) for r in range(rows) for c in range(cols)]
    probs, _, ood = pred.analyse([frame[y0:y1, x0:x1] for y0, x0, y1, x1 in boxes])
    return (boxes, probs, ood) if with_ood else (boxes, probs)


def process(frame, pred, rows, cols, thr):
    """Annotates frame in place; returns fraction of tiles classified as weed."""
    boxes, probs, ood = detect_tiles(frame, pred, rows, cols, with_ood=True)
    overlay = frame.copy()
    weed_tiles = 0
    for (y0, x0, y1, x1), p, unknown in zip(boxes, probs, ood):
        if unknown:  # not a field/plant tile -> never flagged as weed
            cv2.putText(frame, "unknown", (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (160, 160, 160), 1, cv2.LINE_AA)
            continue
        k = int(p.argmax()); name = pred.classes[k]
        if is_weed(name) and p[k] >= thr:
            weed_tiles += 1
            col = pred.colors[k]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), col, -1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
            cv2.putText(frame, f"{name} {p[k]:.2f}", (x0 + 4, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{name} {p[k]:.2f}", (x0 + 4, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    return weed_tiles / len(boxes)


def hud(frame, fps, coverage, kind):
    y = frame.shape[0] - 56  # bottom-left, so it never hides a tile label
    cv2.rectangle(frame, (0, y), (330, y + 56), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS {fps:5.1f}  [{kind}]", (8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Weed coverage {coverage*100:5.1f}%", (8, y + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help=".pt checkpoint or .onnx model")
    p.add_argument("--source", default="0", help="webcam index, video path or image path")
    p.add_argument("--grid", type=int, nargs=2, default=[3, 4], metavar=("ROWS", "COLS"))
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--save", help="optional output video/image path")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    pred = Predictor(a.model, a.threads)
    rows, cols = a.grid
    if a.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = cv2.imread(a.source)
        t = time.perf_counter(); cov = process(img, pred, rows, cols, a.thr)
        hud(img, 1 / (time.perf_counter() - t), cov, pred.kind)
        cv2.imwrite(a.save or "detection.jpg", img)
        print(f"weed coverage {cov*100:.1f}% -> {a.save or 'detection.jpg'}")
        if not a.no_show:
            cv2.imshow("Weed detection", img); cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {a.source}")
    writer, fps = None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        cov = process(frame, pred, rows, cols, a.thr)
        fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t, 1e-6) if fps else 1 / max(time.perf_counter() - t, 1e-6)
        hud(frame, fps, cov, pred.kind)
        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if not a.no_show:
            cv2.imshow("Weed detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
