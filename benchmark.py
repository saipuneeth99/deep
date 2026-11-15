"""Benchmark latency / FPS / size of all architectures (or exported ONNX models) on this device.

    python benchmark.py                       # compare all archs (random weights, PyTorch CPU)
    python benchmark.py --onnx runs/mobilenet_v3_small/model.onnx runs/mobilenet_v3_small/model.int8.onnx
"""
import argparse, json, os, time
import numpy as np
import torch
from weeds.models import ARCHS, build_model, count_params


def bench(fn, x, warmup=10, iters=50):
    for _ in range(warmup):
        fn(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t)
    ms = np.median(ts) * 1000
    return ms, 1000 / ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", nargs="*")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--save", help="write results to this JSON file")
    a = p.parse_args()
    results = []
    torch.set_num_threads(a.threads)
    x = np.random.randn(1, 3, a.img_size, a.img_size).astype(np.float32)

    print(f"{'model':45s} {'params(M)':>10s} {'size(MB)':>9s} {'latency(ms)':>12s} {'FPS':>7s}")
    if a.onnx:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
        for path in a.onnx:
            s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            name = s.get_inputs()[0].name
            ms, fps = bench(lambda inp: s.run(None, {name: inp}), x)
            print(f"{os.path.relpath(path):45s} {'-':>10s} {os.path.getsize(path)/1e6:9.2f} {ms:12.2f} {fps:7.1f}")
            results.append(dict(model=os.path.relpath(path), size_mb=os.path.getsize(path) / 1e6,
                                latency_ms=ms, fps=fps))
        _save(results, a.save)
        return
    xt = torch.from_numpy(x)
    for arch in ARCHS:
        m = build_model(arch, 9, pretrained=False).eval()
        n = count_params(m)
        with torch.inference_mode():
            ms, fps = bench(m, xt)
        print(f"{arch:45s} {n/1e6:10.2f} {n*4/1e6:9.2f} {ms:12.2f} {fps:7.1f}")
        results.append(dict(model=arch, params_m=n / 1e6, size_mb=n * 4 / 1e6, latency_ms=ms, fps=fps))
    _save(results, a.save)


def _save(results, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump(results, open(path, "w"), indent=2)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
