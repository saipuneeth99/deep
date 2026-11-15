# Project Explained: Real-Time Weed Detection Using Edge AI and Lightweight CNNs

This guide explains what the project does, how every file works, the theory behind each decision, and the questions an examiner is likely to ask (with answers).

---

## 1. The problem in one paragraph

Farmers spray herbicide over the **whole field**, even though weeds cover only a part of it. That wastes chemicals and money and harms the soil. If a camera on a tractor or drone can **recognise weeds in real time**, the sprayer can turn on only over weeds (**spot spraying**). The computer on a tractor is small and has no big GPU and often no internet. So the model must be **small, fast and accurate** and must run **on the device itself (edge AI)**. This project trains lightweight CNNs to recognise 8 weed species, compresses them for edge devices, and runs them in real time on camera frames.

## 2. The pipeline at a glance

```
DeepWeeds images (17,509)
   │  scripts/download_deepweeds.py
   ▼
Stratified split 70/15/15 + augmentation + class-balanced sampling      ← weeds/data.py
   ▼
Lightweight CNN (MobileNetV3 / ShuffleNetV2 / WeedNet-Lite)              ← weeds/models.py
   │  train.py  (transfer learning, AdamW, OneCycle LR, label smoothing, early stopping)
   ▼
best.pt  →  evaluate.py  (accuracy, F1, confusion matrix on unseen test set)
   ▼
export.py  →  model.onnx (FP32)  →  model.int8.onnx (quantised, about 3–4× smaller)
   ▼
benchmark.py (latency, FPS, size)      realtime.py (webcam/video, grid detection)
   ▼
app.py  (Streamlit UI: classify, Grad-CAM, field detection, video, dashboard)
```

`run_all.sh` runs the whole chain for three architectures with one command.

---

## 3. The dataset: DeepWeeds

* **17,509 colour images, 256×256 px**, taken in real fields in northern Australia (Olsen et al., *Scientific Reports*, 2019).
* **9 classes:** Chinee apple, Lantana, Parkinsonia, Parthenium, Prickly acacia, Rubber vine, Siam weed, Snake weed and **Negative** (no target weed).
* **Imbalanced:** Negative has 9,106 images (52%), while each weed has about 1,000–1,100 images. A naive model that always says "Negative" would already be 52% accurate. That is why we report **macro-F1** and use **balanced sampling**.
* The original download server is offline, so `scripts/download_deepweeds.py` pulls the official CC-BY-4.0 mirror from Hugging Face (one parquet file). It writes each image as a `.jpg` and creates `labels.csv` (`Filename, Label, Species`).

---

## 4. File-by-file walkthrough

### `weeds/data.py`: everything about data
| Function | What it does | Why |
|---|---|---|
| `scan_dataset(root)` | Reads `labels.csv` (DeepWeeds) or a `class_name/` folder layout and returns image paths, labels and class names | The same code works for any weed dataset |
| `make_splits()` | `train_test_split` with **`stratify=labels`**, giving 70% train, 15% validation and 15% test, with a fixed seed of 42 | Stratified means every split keeps the same class proportions. The seed makes the split reproducible, so the test images are never seen during training |
| `get_transforms(train=True)` | **Augmentation**: RandomResizedCrop, horizontal and vertical flips, ±20° rotation, ColorJitter, RandomErasing, then ImageNet normalisation | Field images vary in angle, lighting, scale and occlusion. Augmentation creates new variations each epoch, which reduces overfitting |
| `get_transforms(train=False)` | Resize to 255 → CenterCrop 224 → normalise | Evaluation must be deterministic (no randomness) |
| `WeedDataset` | PyTorch `Dataset`: opens the image, converts it to RGB, applies the transform and returns `(tensor, label)` | |
| `make_loaders()` | Builds the DataLoaders. For training it uses a **`WeightedRandomSampler`** with weight = 1/class_count | Each class is sampled about equally often, so the model can't get lazy by predicting "Negative" |

**Normalisation** uses ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` because the pretrained backbones were trained with those statistics.

### `weeds/models.py`: the CNN architectures
* **`DSConv`**: a depthwise-separable convolution block:
  1. **Depthwise 3×3 conv** (`groups=cin`): each channel gets its own 3×3 filter (spatial filtering).
  2. **Squeeze-and-Excitation (SE)**: global-average-pool, then a small bottleneck MLP, then hard-sigmoid. This produces one weight per channel, which says "pay more attention to channels that matter" (channel attention).
  3. **Pointwise 1×1 conv**: mixes channels and changes the channel count.
  4. **Residual connection** when the input and output shapes match, which helps gradients flow.
  Each conv layer is followed by BatchNorm and **Hard-Swish** (`x·ReLU6(x+3)/6`). Hard-Swish is a cheap approximation of Swish that quantises well, and MobileNetV3 uses it.
* **`WeedNetLite`**: our **custom architecture designed from scratch**. It has a stride-2 stem conv, then 9 DSConv blocks (channels 32→64→128→256→512, with downsampling by stride 2), then global average pooling, dropout 0.2 and a linear layer. It has **0.47M parameters (1.9 MB)**, about 3× smaller than MobileNetV3-Small. Weights use Kaiming initialisation.
* **`build_model(arch)`**: returns any of the 6 architectures. For torchvision models it loads **ImageNet-pretrained weights** and replaces the final classifier layer with a new `Linear(in_features, 9)`. This is transfer learning.
* `python -m weeds.models` prints the parameter count of every architecture.

### `weeds/utils.py`
* `get_device()` automatically chooses CUDA (NVIDIA GPU), then **MPS** (Apple-silicon GPU), then CPU.
* `load_checkpoint()` rebuilds the right architecture from the saved `arch` name and loads its weights.

### `train.py`: training
Steps in `main()`:
1. Build the loaders and the model, then print the device, parameter count and split sizes.
2. **Loss:** `CrossEntropyLoss(label_smoothing=0.1)`. Instead of a target of 100% for the true class, the target is 90% for the true class plus 10% spread over the others. This stops the model from becoming over-confident and improves generalisation.
3. **Optimiser:** **AdamW** (Adam with decoupled weight decay 1e-4, which acts as L2 regularisation).
4. **Learning-rate schedule:** **OneCycleLR**. The rate warms up for the first 10% of steps, then follows cosine decay toward zero. This converges faster and more stably than a fixed learning rate.
5. `run_epoch()` runs forward pass → loss → `backward()` → **gradient clipping (max norm 5)** → `optimizer.step()` → `scheduler.step()`. It tracks loss and accuracy. In evaluation mode the same function runs without gradients.
6. After every epoch it validates. If validation accuracy improved, it saves **`best.pt`** (weights, arch, classes, img_size, epoch). This is **model checkpointing**.
7. **Early stopping**: it stops if validation accuracy hasn't improved for `--patience` epochs.
8. It saves `history.json` and `curves.png` (loss and accuracy curves for train vs val).

Example: `python train.py --arch mobilenet_v3_small --epochs 15`

### `evaluate.py`: honest measurement on the unseen test set
It loads `best.pt`, predicts every test image, and computes:
* **Accuracy**: correct / total.
* **Precision** per class: of the images predicted as X, how many really are X.
* **Recall** per class: of the real X images, how many we found.
* **F1** = 2·P·R/(P+R). **Macro-F1** is the plain average over classes, so every weed counts equally despite the imbalance. **Weighted-F1** weights each class by its frequency.
* A **normalised confusion matrix**: row = true class, column = predicted class. The diagonal is per-class recall. Off-diagonal cells show which weeds get confused with each other.

It writes `test_metrics.json`, `test_report.txt` and `confusion_matrix.png`.

### `export.py`: preparing for the edge
1. **ONNX export** (`torch.onnx.export`, opset 17, dynamic batch size). ONNX (Open Neural Network Exchange) is a framework-independent model format. **ONNX Runtime** runs it on Raspberry Pi, Jetson, Android and Windows **without PyTorch**. It can also be converted to TensorRT (Jetson) or TFLite.
2. **FP16 conversion**: the weights are stored as 16-bit floats, which makes the model 2× smaller with no accuracy loss. Inputs and outputs stay FP32.
3. **INT8 static post-training quantisation** (`quantize_static`). This happens in three parts:
   * `CalibReader` feeds **320 training images** through the network, and the **MinMax** calibrator records the range of every activation tensor.
   * Weights are stored as 8-bit integers with **per-channel** scales: `real ≈ scale × (q − zero_point)`.
   * The output is in **QDQ** format (QuantizeLinear/DequantizeLinear pairs), which ONNX Runtime fuses into integer kernels.
4. **Accuracy check**: it evaluates FP32, FP16 and INT8 on 1,000 test images and writes `quant_check.json`, so we can prove whether compression hurt accuracy.
5. It writes `classes.txt` so the ONNX model knows the label names.

> **What we found during development:** *dynamic* INT8 quantisation (activation ranges guessed at runtime) collapsed MobileNetV3 to predicting "Negative" for every image (52%). Static calibration helped, but MobileNetV3 still dropped to 61%. Our WeedNet-Lite and ShuffleNetV2 lost **no** accuracy. See section 6.

### `benchmark.py`: is it really real-time?
For each model it does 10 warm-up runs, then times 50 single-image inferences and reports the **median latency (ms)**, **FPS = 1000 / latency**, the parameter count and the size. It uses batch size 1 because a camera produces one frame at a time. The `--onnx` option benchmarks the exported FP32 and INT8 models, and `--save` writes JSON for the dashboard.

### `realtime.py`: the real-time detector
* **`Predictor`**: wraps either a PyTorch `.pt` checkpoint or an ONNX model behind the same interface. `preprocess()` resizes tiles to 224, converts BGR (OpenCV) to RGB, normalises, and changes HWC to CHW. Softmax turns the logits into probabilities.
* **`detect_tiles()`**: splits the frame into an **R×C grid** and classifies **all tiles in a single batched forward pass** (batching is much faster than one tile at a time).
* **`process()`**: for every tile whose top class is a weed with confidence ≥ threshold, it draws a coloured box and the label, and it computes the **weed coverage %** (weed tiles / all tiles). That number is what a sprayer controller would use.
* **`hud()`**: overlays FPS and coverage on the frame.
* `main()` accepts a webcam index (`--source 0`), a video file or an image, and can save the output video.

> **Why grid classification instead of YOLO-style detection?** DeepWeeds has **image-level labels only (no bounding boxes)**. Classifying tiles turns a classifier into a coarse localiser without needing box annotations, and it matches how spray nozzles work: each nozzle covers a fixed strip or cell of the ground. Moving to YOLO or SSD would need box-annotated data (future work).

### `weeds/gradcam.py`: explainability
**Grad-CAM** (Selvaraju et al., 2017) works like this:
1. Hook the last convolutional block and record its **activations A** (feature maps) during the forward pass.
2. Backpropagate the score of the predicted class and record the **gradients dY/dA**.
3. Average the gradients over space to get one **importance weight per channel**.
4. heatmap = ReLU(Σ weight_k · A_k), upsample it to the image size and normalise it.

Red regions are the pixels that pushed the model toward its decision. This shows the model looks at the **plant's leaves and stems, not the background or soil**, which supports the claim that it learned meaningful features.

### `weeds/ood.py` + `calibrate_ood.py`: rejecting images that aren't field photos
**The problem:** a classifier trained on 9 classes *must* pick one of them. Softmax spreads 100% across the 9 classes even for a photo of a classroom. In our first demo, a classroom photo was labelled "WEED: Chinee apple, 41%".

**The fix: deep k-nearest-neighbour out-of-distribution (OOD) detection** (Sun et al., ICML 2022):
1. `WithFeatures` wraps the CNN so it also returns the **penultimate feature vector**, the input to the last linear layer (1,024-dimensional for MobileNetV3). The ONNX models export it as a second output, `features`.
2. `calibrate_ood.py` stores L2-normalised features of **4,000 training images** (the "memory bank"). It then computes, for every validation image, the cosine distance to its **5th nearest neighbour** in the bank. The **99th percentile** of those distances becomes the threshold (97% in v1), so 99% of genuine field images are accepted. This is saved in `runs/<arch>/ood.npz`.
3. At inference, `Predictor.analyse()` computes the same distance. If it is above the threshold, the input or tile is **rejected as "not a field image"** and never marked as a weed.

**Measured results** (MobileNetV3-Small, threshold 0.473):

| Input | Distance | Result |
|---|---:|---|
| Real test images | median 0.21 | 96.8% accepted |
| Classroom photo | 0.530 | rejected ✅ |
| Face close-up | 0.504 | rejected ✅ |
| White wall | 0.523 | rejected ✅ |
| App screenshot | 0.605 | rejected ✅ |
| Plain grey | 0.581 | rejected ✅ |

**Finding:** the ImageNet-pretrained MobileNetV3 separates non-field images much better than WeedNet-Lite, which was trained from scratch. WeedNet-Lite let the classroom photo through, though it labelled it "Negative", never as a weed. Its features only ever learned weeds versus soil, while pretrained features "know" what people and rooms look like.

**Trade-off:** a stricter filter also rejects some real weeds. On the demo video, MobileNetV3 found 69/69 weed tiles with the filter off, 61/69 at a 95% cut-off, and 65/69 at 97%. The sidebar slider lets you tune this live. In v2 the trained outlier class does most of the rejection, so the default is 99% (see section 6b).

### `app.py`: the Streamlit front-end (`streamlit run app.py`)
* **Sidebar:** choose a trained model and the inference engine (**PyTorch FP32 / ONNX FP32 / ONNX INT8**), and see the model size, test accuracy, macro-F1 and confidence threshold.
* **🔍 Classify image:** use a random unseen **test** image (with its ground truth shown), an upload or a **camera snapshot**. The tab shows the prediction, confidence, latency/FPS, a probability bar chart and the **Grad-CAM heatmap**.
* **🌾 Field detection:** builds a *synthetic field* by stitching random test images into a grid (about half weeds), or takes an uploaded field photo. It runs grid detection and shows the annotated image, coverage %, and **tile accuracy against the ground truth** with a per-tile table.
* **🎥 Video / real-time:** upload a video. Frames are processed and displayed live with FPS and coverage, followed by a coverage-over-time chart.
* **📊 Dashboard:** compares all trained models (parameters, validation and test accuracy, macro-F1, CPU latency, ONNX FP32 and INT8 size and latency), shows an **accuracy-versus-latency scatter plot** (bubble size = parameters), and displays the learning curves, confusion matrix and classification report.
* `@st.cache_resource` loads each model only once, which keeps the app fast.

### Other files
* `run_all.sh`: the full pipeline (download → train → evaluate → export for 3 models → benchmark).
* `scripts/make_demo_video.py`: builds `data/demo_field.mp4`, a camera "driving" over a field made of unseen test images. Use it in the Video tab or with `realtime.py`.
* `scripts/make_synthetic.py`: a tiny fake dataset to smoke-test the code without downloading anything.
* `requirements.txt`: the dependencies.
* `runs/<arch>/`: all the outputs for each model.

---

## 5. Key concepts you must be able to explain

**CNN.** Convolution layers slide small learnable filters over the image. Early layers detect edges and colours, deeper layers detect leaf shapes and textures. Pooling and strides shrink the spatial size, and a final linear layer outputs one score per class.

**Why depthwise-separable convolutions are cheap.** For a feature map of size D_F×D_F with M input channels, N output channels and a K×K kernel:
* Standard conv cost: K²·M·N·D_F²
* Depthwise + pointwise cost: K²·M·D_F² + M·N·D_F²
* Ratio = 1/N + 1/K². For K=3 that is about **1/8 to 1/9**, so roughly 8–9× fewer multiply-adds for almost the same accuracy. This is the core idea behind MobileNet, and it is what makes these networks lightweight.

**ShuffleNetV2** uses group convolutions plus *channel shuffle* so that information mixes between groups, and it is designed for real measured speed (memory access cost), not just FLOPs.

**Transfer learning.** The backbone already learned general visual features (edges, textures, shapes) from 1.2M ImageNet images. We replace only the last layer and fine-tune everything, so we get high accuracy with less data and far fewer epochs. WeedNet-Lite is trained **from scratch** for comparison, which shows how much pretraining helps.

**Overfitting and its controls.** Data augmentation, dropout, weight decay, label smoothing, early stopping, and a separate validation set used for model selection. The test set is used **only once**, at the end.

**Edge AI.** Inference runs on the device instead of the cloud. The benefits are low latency, no internet needed in the field, privacy, and low power and cost. The constraints are limited memory, compute and battery, which is why we track parameters, MB, ms and FPS alongside accuracy.

**Quantisation.** FP32 to INT8 makes the model about 4× smaller, uses less memory bandwidth, and allows integer arithmetic on CPUs, NPUs and DSPs. *Dynamic* quantisation (used here) quantises weights offline and activations at runtime and needs no calibration data. *Static* quantisation also pre-computes activation ranges from calibration images and is usually faster on ARM. *Quantisation-aware training (QAT)* simulates quantisation during training to recover the lost accuracy.

---

## 6. Results (DeepWeeds, held-out test set of 2,627 images)

Training was done on an Apple-silicon Mac GPU (MPS). Latency is measured on the Mac CPU with 4 threads and batch size 1.

| Model | Params | Epochs | Test acc | Macro-F1 | PyTorch CPU | ONNX FP32 | ONNX INT8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MobileNetV3-Small** (ImageNet-pretrained) | 1.53M | 15 | **93.3%** | **92.0%** | 28.7 ms | 3.2 ms (6.1 MB) | 1.0 ms (1.9 MB), **acc → 60.7% ❌** |
| **WeedNet-Lite** (ours, from scratch) | **0.47M** | 30 | 88.2% | 86.2% | **7.6 ms** | 2.7 ms (1.9 MB) | 1.8 ms (**0.68 MB**), acc 88.4% ✅ |
| ShuffleNetV2 (pretrained) | 1.26M | 6 | 88.1% | 86.8% | 11.3 ms | 2.5 ms (5.1 MB) | 1.7 ms (1.6 MB), acc 87.8% ✅ |

Compression accuracy check (1,000 test images, ONNX Runtime):

| Model | FP32 | FP16 | INT8 static | INT8 size vs FP32 |
|---|---:|---:|---:|---:|
| MobileNetV3-Small | 92.6% | 92.6% | 60.7% | 3.3× smaller |
| WeedNet-Lite | 88.2% | 88.3% | 88.4% | 2.8× smaller |
| ShuffleNetV2 | 87.9% | 87.8% | 87.8% | 3.1× smaller |

**Per-class results for MobileNetV3-Small:** every weed species has **recall of 88–99%**. Snake weed is the hardest (F1 0.87). The most common errors are *Negative images predicted as a weed* (Negative→Parthenium 20, →Prickly acacia 19, →Siam weed 18). This comes from class-balanced sampling, which favours finding weeds (high recall) over avoiding false alarms. For spot spraying that is the safer trade-off: a false positive wastes a little herbicide, while a missed weed keeps spreading.

**Key findings to state in the viva**
1. **Transfer learning wins on accuracy.** Pretrained MobileNetV3-Small reaches 93.3% in only 15 epochs.
2. **Our custom WeedNet-Lite is the best size/speed trade-off.** It has 3.3× fewer parameters than MobileNetV3-Small and is about 3.8× faster in PyTorch on CPU. It reaches 88.2% **trained from scratch**, and its INT8 model is only **0.68 MB with zero accuracy loss**.
3. **Edge compression is model-dependent.** FP16 is always safe (2× smaller). Post-training INT8 kept full accuracy for WeedNet-Lite and ShuffleNetV2 but **broke the ImageNet-pretrained MobileNetV3**. WeedNet-Lite also uses Hard-Swish and SE, so those layers alone are not the cause. The evidence points to MobileNetV3's pretrained depthwise weights (see Q12b). The fix would be quantisation-aware training, or simply using FP16 for that model.
4. **Every model is real-time.** Even the slowest path (PyTorch on CPU, about 29 ms) is above 30 FPS, and ONNX INT8 reaches 500–1000 FPS on a laptop CPU. A Raspberry Pi is roughly 10–20× slower than this machine, so a 3×4 grid at a few FPS up to tens of FPS is realistic there (to be verified on the device).

All the raw numbers are in `runs/<arch>/test_metrics.json`, `quant_check.json`, `test_report.txt` and `runs/benchmark.json`. The plots are `curves.png` and `confusion_matrix.png`, and the app's Dashboard tab shows them all.

---

## 6b. Version 2: outlier class, longer training, stronger augmentation

**What changed from v1**
1. **A 10th class, "Not a field image" (outlier exposure; Hendrycks et al., ICLR 2019).** It was trained on 2,378 non-field photos: 1,699 from COCO val2017 (people, faces, objects, streets) and 679 from MIT Indoor Scenes (65 room types, including classrooms, offices and labs). `scripts/download_outliers.py` downloads them and drops any image where green vegetation covers more than 15% of pixels (Excess-Green index), so the class never learns that "plants = not a field". Your own photos can be added to `data/outliers_custom/`.
2. **Longer training.** MobileNetV3 went from 15 to 35 epochs, ShuffleNetV2 from 6 to 30, and WeedNet-Lite from 30 to 50.
3. **Stronger augmentation.** MixUp or CutMix is applied to half the batches; it blends two images and their labels so the model learns smoother decision boundaries. Gaussian blur, random greyscale and stronger colour jitter were added too.
4. **The DeepWeeds split is identical to v1.** Outlier images are split separately and appended, so the v1 and v2 numbers are measured on exactly the same 2,627 DeepWeeds test images.

**Results (test set: 2,627 DeepWeeds images + 357 non-field images)**

| Model | DeepWeeds acc v1 → v2 | DeepWeeds macro-F1 v1 → v2 | Non-field images rejected | Field images wrongly called "not a field" |
|---|---|---|---:|---:|
| MobileNetV3-Small | 93.3% → 93.2% | 92.0% → 92.1% | **100%** (357/357) | 0% |
| ShuffleNetV2 | 88.1% → **90.9%** | 86.8% → 89.9% | 99.7% | 0% |
| WeedNet-Lite (ours) | 88.2% → **89.2%** | 86.2% → 88.0% | 99.7% | 0% |

Edge formats in v2 (1,000 test images): FP16 = FP32 for every model. INT8 is still lossless for ShuffleNetV2 (92.6%) and WeedNet-Lite (90.3%, **0.70 MB**), and still broken for MobileNetV3 (37%), so deploy MobileNetV3 as FP16.

**Your classroom photo, after v2:** every model rejects it, along with the face, white wall, grey image and app screenshot. In v1, WeedNet-Lite and ShuffleNetV2 accepted the classroom as "Negative". Its feature distance rose from 0.53 to 0.86 for MobileNetV3, so the outlier class pushed non-field images far away from field images in feature space.

**Defaults retuned for v2.** The kNN filter now accepts 99% of real images (v1: 97%), because the trained outlier class does most of the rejection. The weed confidence threshold is now 0.5 (v1: 0.6), because MixUp and label smoothing make the model less over-confident. On the demo video, all three models find **65/69 weed tiles**.

**Why MobileNetV3 accuracy didn't rise:** it was already near its ceiling at 93% after 15 epochs, and it now also has to learn the 10th class. Its gain is the 100% outlier rejection. The from-scratch and lightly trained models (WeedNet-Lite, ShuffleNetV2) gained the most from longer training.

**Not done (deliberately):** merging other weed datasets. Their species differ from DeepWeeds, so the task would change and the results would no longer be comparable to the published DeepWeeds numbers.

## 7. Likely viva questions and answers

**Q1. Why lightweight CNNs and not ResNet-50 or ViT?**
ResNet-50 has about 25M parameters (about 100 MB) and roughly 4 GFLOPs, which is too slow for real time on a Raspberry Pi. Our models have 0.5–1.5M parameters and run at tens to hundreds of FPS on a CPU while staying accurate. The goal is the best **accuracy per millisecond**.

**Q2. What is edge AI and why does it matter here?**
Running the model on the device in the field (tractor or drone computer). Fields have poor connectivity, and a sprayer moving at about 2 m/s needs decisions within milliseconds, so a round trip to the cloud is not acceptable.

**Q3. Explain depthwise-separable convolution.**
See section 5. It splits spatial filtering (depthwise) from channel mixing (pointwise) and costs about 1/N + 1/K² of a standard conv.

**Q4. What is the role of SE blocks?**
Channel attention. They squeeze each channel to one number (global average pooling), learn per-channel importance weights with a tiny MLP, and rescale the channels. This adds accuracy at very low cost.

**Q5. Your dataset is imbalanced. How did you handle it?**
Stratified splits, `WeightedRandomSampler` (inverse-frequency sampling), label smoothing, and reporting **macro-F1** in addition to accuracy, because accuracy alone is misleading when 52% of images are Negative.

**Q6. Why a separate validation and test set?**
Validation is used to select the best epoch and tune settings. Because those choices are fitted to it, its score is optimistic. The test set is untouched until the end, so it gives an unbiased estimate.

**Q7. What augmentations did you use and why?**
Random resized crop (scale and position), horizontal and vertical flips (plants have no fixed orientation from a top-down camera), rotation, colour jitter (sun versus shade), and random erasing (occlusion by other plants). Together they make the model robust to field conditions.

**Q8. What is label smoothing?**
It softens the one-hot target to 0.9 for the true class plus 0.1 spread over the others. This prevents over-confident logits and gives better calibration and generalisation.

**Q9. Why AdamW and OneCycle?**
AdamW adapts the learning rate per parameter and applies weight decay correctly (decoupled from the gradient). OneCycle warms up, then anneals, which gives fast, stable convergence and usually better final accuracy.

**Q10. How do you know the model isn't just looking at the background?**
Grad-CAM heatmaps highlight the plant's leaves and structures. The confusion matrix also shows the errors are between visually similar species, not random.

**Q11. What is ONNX and why export to it?**
A standard, framework-independent model format. ONNX Runtime is a lightweight inference engine for ARM, x86 and GPUs, so the edge device doesn't need PyTorch. It can also be converted to TensorRT or TFLite.

**Q12. What does INT8 quantisation do? Any downside?**
It stores weights (and, with static quantisation, activations) as 8-bit integers with a scale and zero-point, making the model about 3–4× smaller and faster. In our runs the INT8 latency dropped from 3.2 ms to 1.0 ms for MobileNetV3. The downside is possible accuracy loss, and it depends on the architecture. WeedNet-Lite and ShuffleNetV2 lost nothing, but MobileNetV3-Small fell from 92.6% to 60.7%. Dynamic quantisation was even worse (52%, everything predicted as Negative).

**Q12b. Why did MobileNetV3 fail INT8 while WeedNet-Lite didn't?**
It wasn't the layer types: WeedNet-Lite also uses Hard-Swish and SE blocks and lost no accuracy. The difference is in the **learned weights and activations**. The ImageNet-pretrained MobileNetV3 has depthwise channels whose weight ranges differ enormously, and some are almost zero. During quantisation ONNX Runtime warned that bias values overflowed because a channel's scale was nearly zero. Its activations also have large outliers, so with only 256 levels the small but important values round to zero, and errors accumulate across about 50 layers. This is a known problem with quantising pretrained MobileNet-family models after training. We tried per-tensor scales, symmetric activations, float biases, and quantising only the convolutions; none of them got above about 61%. Two fixes would work: quantisation-aware training (QAT), or deploying MobileNetV3 in FP16, which halves the size with no loss. Our WeedNet-Lite was trained from scratch with weight decay, which keeps its weights in compact, quantisation-friendly ranges.

**Q12c. Static versus dynamic quantisation?**
Dynamic quantisation converts weights offline and computes activation ranges at runtime for each input. That is easy (no data needed) but inaccurate for CNNs. Static quantisation runs calibration images through the network first to fix the activation ranges, and it is the standard for CNNs. We used static quantisation with 320 calibration images, per-channel weight scales and the QDQ format.

**Q13. How do you do *detection* with a classifier?**
Grid-tile classification. The frame is split into R×C cells, all cells are classified in one batched forward pass, and the weed cells are marked. This works because DeepWeeds has image-level labels only, and it maps directly onto spray-nozzle zones. A finer grid gives better localisation but each tile has less context and inference is slower.

**Q14. Why not YOLO?**
YOLO needs bounding-box annotations, and DeepWeeds doesn't have them. With a box-annotated dataset, a lightweight detector (YOLOv8n, or MobileNet-SSD) would be the natural extension.

**Q15. What's the difference between latency and throughput?**
Latency is the time for one frame (ms). Throughput is how many frames per second can be processed. We benchmark with batch size 1 because a live camera produces frames one by one.

**Q16. Why measure the median, with warm-up runs?**
The first runs include memory allocation and kernel compilation, so they are slow. The median is robust to occasional OS hiccups.

**Q17. What's the difference between macro and weighted F1?**
Macro averages the per-class F1 equally (rare classes count fully). Weighted averages by class frequency (dominated by Negative here).

**Q17b. What accuracy did you get?**
MobileNetV3-Small reached 93.3% test accuracy with a macro-F1 of 92.0%. Our WeedNet-Lite reached 88.2% with only 0.47M parameters, trained from scratch, and kept 88.4% after INT8 at 0.68 MB. For reference, the original DeepWeeds paper reported about 95.7% with ResNet-50 (25M parameters), so we are within about 2.5 points of it with a model roughly 16× smaller.

**Q18. What is the custom contribution of this project?**
(1) WeedNet-Lite, a custom 0.47M-parameter SE-augmented depthwise-separable CNN. (2) A systematic comparison of accuracy versus latency versus size across lightweight architectures on DeepWeeds. (3) A full edge pipeline: ONNX, INT8, and real-time grid detection with a weed-coverage output. (4) Explainability with Grad-CAM. (5) An interactive Streamlit demo.

**Q19. What does the confusion matrix show?**
For MobileNetV3, every weed species has a recall of 88–99%. The most common errors are **Negative images predicted as weeds** (for example 20 Negative→Parthenium and 19 Negative→Prickly acacia), and Snake weed is the weakest class (F1 0.87). Negative contains all kinds of other plants that can look like weeds. Balanced sampling deliberately tilts the model toward finding weeds: weed recall is high but weed precision is about 0.85–0.93. For spraying that's the right trade-off.

**Q20. What is BatchNorm and why use it?**
It normalises each channel's activations over the batch, then learns a scale and shift. This stabilises and speeds up training and allows higher learning rates. At inference it uses running statistics and can be fused into the conv.

**Q21. What is Hard-Swish?**
`x · ReLU6(x+3)/6`, a piecewise-linear approximation of Swish (`x·sigmoid(x)`). It gives Swish-like accuracy without the expensive exponential and is friendly to quantisation.

**Q22. Overfitting: how would you see it?**
Training accuracy keeps rising while validation accuracy plateaus or falls, and the validation loss rises (check `curves.png`). Early stopping takes the best validation epoch.

**Q23. How would you deploy this on a Raspberry Pi?**
`pip install onnxruntime opencv-python numpy`, copy `model.int8.onnx` and `classes.txt`, then run `python realtime.py --model model.int8.onnx --source 0`. For Jetson, convert with TensorRT FP16.

**Q24. Limitations?**
Grid localisation is coarse. DeepWeeds comes from one region (Australian rangelands), so performance can drop under domain shift (other crops, soil or lighting). The Negative class is broad. INT8 speed depends on the hardware.

**Q24b. If I show your app a photo of my face, why doesn't it say "weed"?**
A plain classifier would, because it always picks one of the 9 classes. Our app first checks whether the image is *out-of-distribution*. It compares the CNN's feature vector with 4,000 real training images (deep k-nearest-neighbour distance). In v2 there is also a trained 10th class, "Not a field image", learned from 2,378 photos of people, rooms and objects. If either check fires, the app says "Not a field / plant image" and makes no decision. A confidence threshold alone is not enough, because neural networks are often over-confident on unfamiliar inputs. Images that pass that check but have low confidence are shown as "Uncertain".

**Q25. Future work?**
Box- or mask-annotated data with YOLOv8n / segmentation; static INT8 or QAT; knowledge distillation from a large teacher into WeedNet-Lite; testing on a Raspberry Pi or Jetson with power measurements; a multi-spectral camera; driving actual spray nozzles.

---

## 8. How to run everything

```bash
source .venv/bin/activate
bash run_all.sh                 # download + train 3 models + evaluate + export + benchmark
streamlit run app.py            # open the UI (http://localhost:8501)
python realtime.py --model runs/mobilenet_v3_small/model.onnx --source 0   # live webcam window
```
