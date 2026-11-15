"""Export a checkpoint for edge deployment: ONNX FP32, FP16 and INT8 (+ accuracy check of each).

* FP16: weights stored as half precision -> 2x smaller, ~no accuracy loss, fast on GPUs/NPUs.
* INT8: *static* post-training quantization. A few hundred training images calibrate the
  activation ranges (MinMax) and weights get per-channel scales -> ~4x smaller.
  Dynamic quantization was tried first and collapsed MobileNetV3 to predicting a single class.
  Some models (here: the ImageNet-pretrained MobileNetV3) lose a lot of accuracy under post-training
  INT8 and need quantization-aware training; quant_check.json records this per model.

ONNX models run on Raspberry Pi / Jetson / mobile via ONNX Runtime, and can be
converted to TensorRT (Jetson) or TFLite (via onnx2tf) if needed.
"""
import argparse, json, os
import numpy as np
import torch
from weeds.utils import load_checkpoint


class CalibReader:
    """Feeds preprocessed training images to the ONNX Runtime calibrator."""

    def __init__(self, paths, img_size, input_name, batch=16):
        from PIL import Image
        from weeds.data import get_transforms
        tf = get_transforms(img_size, train=False)
        imgs = [tf(Image.open(p).convert("RGB")).numpy() for p in paths]
        self.batches = iter([{input_name: np.stack(imgs[i:i + batch])} for i in range(0, len(imgs), batch)])

    def get_next(self):
        return next(self.batches, None)


def onnx_accuracy(path, paths, labels, img_size):
    import onnxruntime as ort
    from PIL import Image
    from weeds.data import get_transforms
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    tf = get_transforms(img_size, train=False)
    correct = 0
    for i in range(0, len(paths), 50):
        x = np.stack([tf(Image.open(p).convert("RGB")).numpy() for p in paths[i:i + 50]])
        correct += (s.run(None, {name: x})[0].argmax(1) == np.array(labels[i:i + 50])).sum()
    return correct / len(paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/deepweeds", help="used for INT8 calibration + accuracy check")
    p.add_argument("--calib", type=int, default=320, help="number of calibration images")
    p.add_argument("--check", type=int, default=1000, help="test images for FP32 vs INT8 accuracy check")
    p.add_argument("--no-int8", action="store_true")
    a = p.parse_args()

    model, ckpt = load_checkpoint(a.ckpt, "cpu")
    out_dir = os.path.dirname(a.ckpt)
    fp32 = os.path.join(out_dir, "model.onnx")
    from weeds.ood import WithFeatures
    dummy = torch.randn(1, 3, ckpt["img_size"], ckpt["img_size"])
    # second output = penultimate features, used for out-of-distribution rejection (weeds/ood.py)
    torch.onnx.export(WithFeatures(model).eval(), dummy, fp32, input_names=["input"],
                      output_names=["logits", "features"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"FP32 ONNX -> {fp32} ({os.path.getsize(fp32)/1e6:.2f} MB)")
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(ckpt["classes"]))
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    fp16 = os.path.join(out_dir, "model.fp16.onnx")
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), fp16)
    print(f"FP16 ONNX -> {fp16} ({os.path.getsize(fp16)/1e6:.2f} MB)")
    if a.no_int8:
        return

    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from weeds.data import splits_for
    (trp, _), _, (tep, tel), _ = splits_for(ckpt, a.data)
    rng = np.random.default_rng(0)
    calib_paths = [trp[i] for i in rng.choice(len(trp), min(a.calib, len(trp)), replace=False)]

    pre = os.path.join(out_dir, "model.pre.onnx")
    int8 = os.path.join(out_dir, "model.int8.onnx")
    quant_pre_process(fp32, pre)  # shape inference + graph optimisation (folds BatchNorm etc.)
    quantize_static(pre, int8, CalibReader(calib_paths, ckpt["img_size"], "input"),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)  # Percentile segfaults in ORT 1.19 on macOS
    os.remove(pre)
    print(f"INT8 ONNX -> {int8} ({os.path.getsize(int8)/1e6:.2f} MB)  [static, {len(calib_paths)} calib images]")

    if a.check:
        ids = rng.choice(len(tep), min(a.check, len(tep)), replace=False)
        cp, cl = [tep[i] for i in ids], [tel[i] for i in ids]
        res = {"n_images": len(cp)}
        for tag, path in (("fp32", fp32), ("fp16", fp16), ("int8", int8)):
            res[f"{tag}_acc"] = float(onnx_accuracy(path, cp, cl, ckpt["img_size"]))
            res[f"{tag}_mb"] = os.path.getsize(path) / 1e6
        json.dump(res, open(os.path.join(out_dir, "quant_check.json"), "w"), indent=2)
        print(f"Accuracy on {len(cp)} test images: FP32 {res['fp32_acc']:.4f}  "
              f"FP16 {res['fp16_acc']:.4f}  INT8 {res['int8_acc']:.4f}")


if __name__ == "__main__":
    main()
