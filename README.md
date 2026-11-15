# Real-Time Weed Detection Using Edge AI and Lightweight CNNs

A complete deep-learning pipeline that recognises and localises weeds in field images in real time on low-power edge hardware (Raspberry Pi, Jetson Nano, phones, laptops).

## Overview

| Stage | File | What it does |
|---|---|---|
| Data | `weeds/data.py` | DeepWeeds / ImageFolder loading, stratified 70/15/15 split, augmentation, class-balanced sampling |
| Models | `weeds/models.py` | **WeedNet-Lite** (custom 0.47M-param CNN) + MobileNetV3-S/L, EfficientNet-B0, ShuffleNetV2, SqueezeNet |
| Training | `train.py` | Transfer learning, AdamW + OneCycle LR, label smoothing, grad clipping, early stopping, learning curves |
| Evaluation | `evaluate.py` | Accuracy, macro/weighted F1, per-class report, confusion matrix |
| Edge export | `export.py` | ONNX FP32, FP16 and **static INT8** (calibrated) + an accuracy check of each |
| Benchmark | `benchmark.py` | Params, model size, latency, FPS |
| Real-time | `realtime.py` | Webcam/video/image → grid-based weed localisation with FPS and weed-coverage HUD |
| OOD rejection | `weeds/ood.py`, `calibrate_ood.py` | Rejects inputs that are not field or plant images (people, rooms, screens) using deep-kNN feature distance |
| UI | `app.py` | Streamlit front-end (classification, Grad-CAM, field detection, video, dashboard) |
| Explainability | `weeds/gradcam.py` | Grad-CAM heatmaps |

### WeedNet-Lite (custom architecture)
Stem conv → 9 × depthwise-separable blocks (3×3 depthwise → Squeeze-and-Excitation → 1×1 pointwise, Hard-Swish, residuals when shapes match) → global average pooling → dropout → linear layer.
Depthwise-separable convolutions use roughly 8–9× fewer FLOPs than standard convolutions, and SE attention adds channel re-weighting at very little cost.

### Real-time detection approach
Each frame is split into an R×C grid, and **all tiles are classified in a single batched forward pass**. Tiles predicted as a weed species above the confidence threshold are highlighted with the species name, and the HUD shows FPS and the percentage of the field covered by weeds. The output can drive targeted (spot) herbicide spraying.

## Dataset — DeepWeeds
17,509 RGB images (256×256) of 8 Australian weed species plus a *Negative* class (Olsen et al., *Scientific Reports* 2019).
Classes: Chinee apple, Lantana, Parkinsonia, Parthenium, Prickly acacia, Rubber vine, Siam weed, Snake weed, Negative.

Any other dataset in `root/<class_name>/*.jpg` layout also works (for example Plant Seedlings or crop-vs-weed datasets).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Get data
python scripts/download_deepweeds.py data/deepweeds   # from the Hugging Face mirror (original host is offline)
#   or run everything at once: bash run_all.sh
#   (offline smoke test instead: python scripts/make_synthetic.py data/synthetic)

# 2. Train (try several archs to compare)
python train.py --data data/deepweeds --arch mobilenet_v3_small --epochs 30
python train.py --data data/deepweeds --arch weednet_lite --epochs 60 --no-pretrained --lr 3e-3

# 3. Evaluate on the test split
python evaluate.py --ckpt runs/mobilenet_v3_small/best.pt --data data/deepweeds

# 4. Export for edge (FP32 + INT8 ONNX)
python export.py --ckpt runs/mobilenet_v3_small/best.pt

# 5. Benchmark
python benchmark.py                                    # all architectures
python benchmark.py --onnx runs/mobilenet_v3_small/model.onnx runs/mobilenet_v3_small/model.int8.onnx

# 6. Real-time detection
python realtime.py --model runs/mobilenet_v3_small/model.int8.onnx --source 0          # webcam
python realtime.py --model runs/mobilenet_v3_small/model.onnx --source field.mp4 --save out.mp4
python realtime.py --model runs/mobilenet_v3_small/best.pt --source img.jpg --grid 4 4
```

Each run writes these files to `runs/<arch>/`: `best.pt`, `history.json`, `curves.png`, `test_metrics.json`, `test_report.txt`, `confusion_matrix.png`, `model.onnx`, `model.fp16.onnx`, `model.int8.onnx`, `quant_check.json` and `classes.txt`.

## Results (v2: 10 classes, longer training, MixUp/CutMix)

Test set: the same 2,627 DeepWeeds images as v1, plus 357 non-field images.

| Model | Params | DeepWeeds acc (v1 → v2) | Non-field rejected | PyTorch CPU | ONNX INT8 (acc) |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small (35 epochs) | 1.53M | **93.3% → 93.2%** | **100%** | 28.8 ms | 1.0 ms (37% ❌, use FP16) |
| ShuffleNetV2 (30 epochs) | 1.26M | 88.1% → **90.9%** | 99.7% | 11.3 ms | 1.7 ms (92.6% ✅) |
| **WeedNet-Lite (ours, 50 epochs, from scratch)** | **0.47M** | 88.2% → **89.2%** | 99.7% | **7.6 ms** | 1.9 ms, **0.70 MB** (90.3% ✅) |

* The 10th class, "Not a field image", was trained on 2,378 COCO and MIT Indoor photos (`scripts/download_outliers.py`). Together with the kNN feature-distance filter, it makes the app reject people, rooms and screens.
* v1 models and results are kept in `runs_v1/`. Full analysis: [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md), sections 6 and 6b.

## Streamlit app

```bash
.venv/bin/streamlit run app.py     # http://localhost:8501
```
The app has five tabs:
* **Classify image**: shows top-k probabilities, a Grad-CAM heatmap and the latency.
* **Field detection**: grid weed map with coverage % and per-tile accuracy against the ground truth.
* **Video / real-time**: try it with `data/demo_field.mp4` from `scripts/make_demo_video.py`.
* **Model dashboard**: accuracy vs latency vs size, the FP32/FP16/INT8 comparison, curves and confusion matrices.
* **How it works**: a summary of the pipeline.

The sidebar switches between PyTorch, ONNX FP32, ONNX FP16 and ONNX INT8.

## Deploying on edge devices
- **Raspberry Pi 4/5**: run `pip install onnxruntime opencv-python numpy`, copy `model.int8.onnx` and `classes.txt` to the device, then run `realtime.py`. This needs only `onnxruntime`, not PyTorch.
- **Jetson Nano/Orin**: run `trtexec --onnx=model.onnx --fp16` to build a TensorRT engine.
- **Android / TFLite**: run `pip install onnx2tf && onnx2tf -i model.onnx`.

## References
- Olsen A. et al. *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning.* Sci. Rep. 9, 2058 (2019).
- Howard A. et al. *Searching for MobileNetV3.* ICCV 2019.
- Hu J. et al. *Squeeze-and-Excitation Networks.* CVPR 2018.
- Ma N. et al. *ShuffleNet V2.* ECCV 2018.
