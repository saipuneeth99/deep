"""Streamlit front-end for the weed detection project.

    streamlit run app.py
"""
import glob, json, os, tempfile, time

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from realtime import Predictor, detect_tiles, hud, is_weed, process
from weeds.data import make_splits, get_transforms
from weeds.gradcam import grad_cam
from weeds.utils import load_checkpoint

DATA_ROOT = "data/deepweeds"
RUNS = "runs"

st.set_page_config(page_title="Weed Detection · Edge AI", page_icon="🌿", layout="wide")


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading model ...")
def get_predictor(path):
    return Predictor(path, threads=4)


@st.cache_resource(show_spinner=False)
def get_cam_model(ckpt_path):
    return load_checkpoint(ckpt_path, "cpu")


@st.cache_data(show_spinner=False)
def test_split(root):
    _, _, (paths, labels), classes = make_splits(root)
    return paths, labels, classes


def list_runs():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{RUNS}/*/best.pt"))


def pil_to_bgr(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌿 Weed Detection")
st.sidebar.caption("Edge AI · Lightweight CNNs · DeepWeeds")
runs = list_runs()
if not runs:
    st.error("No trained models found in `runs/`. Train one first: `python train.py --arch mobilenet_v3_small`")
    st.stop()

arch = st.sidebar.selectbox("Model", runs, index=runs.index("mobilenet_v3_small") if "mobilenet_v3_small" in runs else 0)
run_dir = os.path.join(RUNS, arch)
backends = {"PyTorch (FP32)": os.path.join(run_dir, "best.pt")}
for label, f in (("ONNX Runtime (FP32)", "model.onnx"), ("ONNX Runtime (FP16)", "model.fp16.onnx"),
                 ("ONNX Runtime (INT8 quantised)", "model.int8.onnx")):
    if os.path.exists(os.path.join(run_dir, f)):
        backends[label] = os.path.join(run_dir, f)
backend = st.sidebar.radio("Inference engine", list(backends))
model_path = backends[backend]
pred = get_predictor(model_path)
st.sidebar.markdown(f"**File:** `{model_path}`  \n**Size:** {os.path.getsize(model_path)/1e6:.2f} MB  \n"
                    f"**Classes:** {len(pred.classes)}")
metrics_file = os.path.join(run_dir, "test_metrics.json")
if os.path.exists(metrics_file):
    m = json.load(open(metrics_file))
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Test acc", f"{m['accuracy']*100:.1f}%")
    c2.metric("Macro F1", f"{m['macro_f1']*100:.1f}%")
qc_file = os.path.join(run_dir, "quant_check.json")
if os.path.exists(qc_file):
    qc = json.load(open(qc_file))
    if "INT8" in backend and qc["fp32_acc"] - qc["int8_acc"] > 0.03:
        st.sidebar.warning(f"INT8 post-training quantisation drops this model's accuracy from "
                           f"{qc['fp32_acc']*100:.1f}% to {qc['int8_acc']*100:.1f}%. See the dashboard.")
thr = st.sidebar.slider("Weed confidence threshold", 0.3, 0.95, 0.5, 0.05)
if pred.ood is not None and "val_dist" in pred.ood:
    acc_rate = st.sidebar.slider("OOD filter: % of real field images accepted", 90.0, 100.0, 99.0, 0.5,
                                 help="Inputs further from the training data than this percentile of genuine "
                                      "validation images are rejected as 'not a field image'. 100 = filter off.")
    pred.ood_threshold = (np.inf if acc_rate >= 100 else float(np.quantile(pred.ood["val_dist"], acc_rate / 100)))

tabs = st.tabs(["🔍 Classify image", "🌾 Field detection", "🎥 Video / real-time", "📊 Model dashboard", "ℹ️ How it works"])

# ---------------------------------------------------------------- 1. classify single image
with tabs[0]:
    st.subheader("Classify a single plant image")
    have_data = os.path.exists(os.path.join(DATA_ROOT, "labels.csv"))
    src = st.radio("Image source", ["Random test-set image", "Upload", "Camera snapshot"] if have_data
                   else ["Upload", "Camera snapshot"], horizontal=True)
    img, truth = None, None
    if src == "Random test-set image":
        paths, labels, classes = test_split(DATA_ROOT)
        if "idx" not in st.session_state or st.button("🎲 New random image"):
            st.session_state.idx = int(np.random.randint(len(paths)))
        img, truth = Image.open(paths[st.session_state.idx]), classes[labels[st.session_state.idx]]
    elif src == "Upload":
        f = st.file_uploader("Upload a plant photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
        img = Image.open(f) if f else None
    else:
        f = st.camera_input("Point the camera at a plant")
        img = Image.open(f) if f else None

    if img is not None:
        bgr = pil_to_bgr(img)
        pred([bgr])  # warm-up
        t = time.perf_counter(); probs, dist, rej = pred.analyse([bgr]); ms = (time.perf_counter() - t) * 1000
        probs, dist, rejected = probs[0], (None if dist is None else float(dist[0])), bool(rej[0])
        k = int(probs.argmax())
        c1, c2, c3 = st.columns([1, 1, 1.3])
        c1.image(img.convert("RGB"), caption="Input", width="stretch")
        if backend.startswith("PyTorch"):
            cam_model, ckpt = get_cam_model(backends["PyTorch (FP32)"])
            x = get_transforms(ckpt["img_size"], False)(img.convert("RGB")).unsqueeze(0)
            cam, _ = grad_cam(cam_model, ckpt["arch"], x, k)
            base = np.array(img.convert("RGB").resize((ckpt["img_size"],) * 2))
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
            c2.image((0.55 * base + 0.45 * heat).astype(np.uint8), caption="Grad-CAM: where the CNN looked",
                     width="stretch")
        else:
            c2.info("Grad-CAM needs gradients, so it is available only with the PyTorch engine.")
        with c3:
            name = pred.classes[k]
            if name == "Not a field image" and probs[k] >= thr:
                st.warning(f"**🚫 Not a field / plant image**  \nThe classifier's outlier class says so "
                           f"({probs[k]*100:.1f}%), so no weed decision is made.")
            elif rejected:
                st.warning("**🚫 Not a field / plant image**  \nThis input is unlike anything in the training data "
                           "(out-of-distribution), so no weed decision is made.")
            elif probs[k] < thr:
                st.info(f"**❔ Uncertain**: best guess {name} at {probs[k]*100:.1f}%, below the "
                        f"{thr*100:.0f}% confidence threshold. No decision is made.")
            else:
                (st.error if is_weed(name) else st.success)(
                    f"**{'🌿 WEED: ' + name if is_weed(name) else '✅ No weed (Negative)'}**  \nconfidence {probs[k]*100:.1f}%")
            if dist is not None:
                thr_ood = pred.ood_threshold
                st.caption(f"OOD distance {dist:.3f} (reject if > {thr_ood:.3f}). This is the distance in CNN feature "
                           f"space to the nearest real DeepWeeds training images.")
            if truth is not None:
                st.write(f"Ground truth: **{truth}** {'✅' if truth == name else '❌'}")
            st.metric("Inference latency", f"{ms:.1f} ms", f"{1000/ms:.0f} FPS", delta_color="off")
            df = pd.DataFrame({"class": pred.classes, "probability": probs})
            st.altair_chart(alt.Chart(df).mark_bar().encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("class:N", sort="-x"),
                color=alt.condition(alt.datum.probability == float(probs.max()), alt.value("#2e7d32"), alt.value("#9e9e9e"))),
                use_container_width=True)

# ---------------------------------------------------------------- 2. field detection (grid)
with tabs[1]:
    st.subheader("Weed localisation on a field image (grid-tile detection)")
    st.caption("The image is split into a grid; every tile is classified in one batched forward pass. "
               "Weed tiles are highlighted, giving a weed map for targeted spraying.")
    c1, c2, c3 = st.columns(3)
    rows = c1.slider("Grid rows", 1, 8, 3)
    cols = c2.slider("Grid cols", 1, 8, 4)
    mode = c3.radio("Field image", ["Synthetic field (mosaic of test images)", "Upload"] if have_data else ["Upload"])
    field, gt = None, None
    if mode.startswith("Synthetic"):
        paths, labels, classes = test_split(DATA_ROOT)
        if st.button("🎲 Generate new field") or "mosaic" not in st.session_state or st.session_state.get("mshape") != (rows, cols):
            rng = np.random.default_rng()
            # ~50% negative tiles, like a real field
            neg = [i for i, l in enumerate(labels) if classes[l] == "Negative"]
            pos = [i for i, l in enumerate(labels) if classes[l] != "Negative"]
            ids = [int(rng.choice(neg if rng.random() < 0.5 else pos)) for _ in range(rows * cols)]
            tiles = [cv2.resize(cv2.imread(paths[i]), (224, 224)) for i in ids]
            st.session_state.mosaic = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
            st.session_state.mgt = [classes[labels[i]] for i in ids]
            st.session_state.mshape = (rows, cols)
        field, gt = st.session_state.mosaic.copy(), st.session_state.mgt
    else:
        f = st.file_uploader("Upload a field photo", type=["jpg", "jpeg", "png"], key="field")
        field = pil_to_bgr(Image.open(f)) if f else None

    if field is not None:
        raw = field.copy()
        detect_tiles(raw, pred, rows, cols)  # warm-up
        t = time.perf_counter(); cov = process(field, pred, rows, cols, thr); ms = (time.perf_counter() - t) * 1000
        hud(field, 1000 / ms, cov, pred.kind)
        a, b = st.columns(2)
        a.image(bgr_to_rgb(raw), caption="Input field", width="stretch")
        b.image(bgr_to_rgb(field), caption="Detected weeds", width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Weed coverage", f"{cov*100:.1f}%")
        m2.metric("Frame time", f"{ms:.1f} ms", f"{rows*cols} tiles")
        if gt is not None:
            _, probs = detect_tiles(raw, pred, rows, cols)
            names = [pred.classes[int(p.argmax())] for p in probs]
            m3.metric("Tile accuracy vs ground truth", f"{np.mean([n == g for n, g in zip(names, gt)])*100:.0f}%")
            st.dataframe(pd.DataFrame({"tile": range(len(gt)), "ground truth": gt, "predicted": names,
                                       "confidence": [f"{p.max():.2f}" for p in probs],
                                       "correct": ["✅" if n == g else "❌" for n, g in zip(names, gt)]}),
                         width="stretch", hide_index=True)

# ---------------------------------------------------------------- 3. video / real-time
with tabs[2]:
    st.subheader("Real-time processing of a video stream")
    st.caption("Upload a field video (or record one with your phone). Frames are processed live and shown "
               "with FPS and weed coverage. For a true live webcam window run: "
               "`python realtime.py --model runs/<arch>/model.int8.onnx --source 0`")
    c1, c2, c3 = st.columns(3)
    vr = c1.slider("Rows", 1, 6, 3, key="vr")
    vc = c2.slider("Cols", 1, 6, 4, key="vc")
    max_frames = c3.slider("Max frames", 30, 600, 200, 10)
    vf = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if vf and st.button("▶️ Run detection"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vf.name)[1])
        tmp.write(vf.read()); tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        view, bar = st.empty(), st.progress(0.0)
        stats = []
        n = 0
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
            t = time.perf_counter(); cov = process(frame, pred, vr, vc, thr); dt = time.perf_counter() - t
            hud(frame, 1 / dt, cov, pred.kind)
            view.image(bgr_to_rgb(frame), width="stretch")
            stats.append(dict(frame=n, fps=1 / dt, coverage=cov * 100))
            n += 1
            bar.progress(n / max_frames)
        cap.release(); os.unlink(tmp.name)
        if stats:
            df = pd.DataFrame(stats)
            s1, s2 = st.columns(2)
            s1.metric("Mean model FPS", f"{df.fps.mean():.1f}")
            s2.metric("Mean weed coverage", f"{df.coverage.mean():.1f}%")
            st.line_chart(df.set_index("frame")[["coverage"]])

# ---------------------------------------------------------------- 4. dashboard
with tabs[3]:
    st.subheader("Model comparison: accuracy versus speed versus size")
    bench = {}
    if os.path.exists(f"{RUNS}/benchmark.json"):
        bench = {r["model"]: r for r in json.load(open(f"{RUNS}/benchmark.json"))}
    rows_ = []
    for r in runs:
        d = os.path.join(RUNS, r)
        tm = json.load(open(f"{d}/test_metrics.json")) if os.path.exists(f"{d}/test_metrics.json") else {}
        ck = torch.load(f"{d}/best.pt", map_location="cpu", weights_only=False)
        row = dict(model=r, params_M=sum(v.numel() for k, v in ck["model"].items()
                                          if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))) / 1e6,
                   best_epoch=ck.get("epoch"), val_acc=ck.get("val_acc"),
                   test_acc=tm.get("accuracy"), macro_f1=tm.get("macro_f1"),
                   deepweeds_acc=tm.get("deepweeds_acc", tm.get("accuracy")),
                   outlier_recall=tm.get("outlier_recall"),
                   cpu_latency_ms=bench.get(r, {}).get("latency_ms"),
                   onnx_mb=os.path.getsize(f"{d}/model.onnx") / 1e6 if os.path.exists(f"{d}/model.onnx") else None,
                   int8_mb=os.path.getsize(f"{d}/model.int8.onnx") / 1e6 if os.path.exists(f"{d}/model.int8.onnx") else None)
        for tag, fn in (("onnx_fp32_ms", "model.onnx"), ("onnx_fp16_ms", "model.fp16.onnx"),
                        ("onnx_int8_ms", "model.int8.onnx")):
            row[tag] = bench.get(f"{d}/{fn}", {}).get("latency_ms")
        qf = f"{d}/quant_check.json"
        if os.path.exists(qf):
            q = json.load(open(qf))
            row.update(fp16_mb=q["fp16_mb"], onnx_fp32_acc=q["fp32_acc"], onnx_fp16_acc=q["fp16_acc"],
                       onnx_int8_acc=q["int8_acc"])
        rows_.append(row)
    df = pd.DataFrame(rows_)
    st.dataframe(df.style.format(precision=3, na_rep="-"), width="stretch", hide_index=True)
    st.caption("test_acc and macro_f1 include the 'Not a field image' class. deepweeds_acc counts only the 2,627 "
               "DeepWeeds test images, so it is comparable with the published DeepWeeds results. outlier_recall is the "
               "share of non-field test images (people, rooms, objects) correctly rejected.")
    plot = df.dropna(subset=["test_acc", "cpu_latency_ms"])
    if len(plot):
        st.altair_chart(alt.Chart(plot).mark_circle(opacity=.8).encode(
            x=alt.X("cpu_latency_ms:Q", title="CPU latency (ms, lower is better)"),
            y=alt.Y("test_acc:Q", title="Test accuracy", scale=alt.Scale(zero=False)),
            size=alt.Size("params_M:Q", title="Params (M)", scale=alt.Scale(range=[100, 1500])),
            color="model:N", tooltip=list(plot.columns)).properties(height=380), use_container_width=True)
    if "onnx_int8_acc" in df:
        st.markdown("#### Edge compression: accuracy of each exported format (1,000 test images)")
        q = df.dropna(subset=["onnx_int8_acc"]).melt(id_vars="model", value_vars=["onnx_fp32_acc", "onnx_fp16_acc", "onnx_int8_acc"],
                                                    var_name="format", value_name="accuracy")
        q["format"] = q["format"].str.replace("onnx_", "").str.replace("_acc", "").str.upper()
        st.altair_chart(alt.Chart(q).mark_bar().encode(
            x=alt.X("format:N", title=None, sort=["FP32", "FP16", "INT8"]), y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("format:N", sort=["FP32", "FP16", "INT8"]), column=alt.Column("model:N", title=None),
            tooltip=["model", "format", alt.Tooltip("accuracy:Q", format=".3f")]).properties(width=140, height=220))
        st.caption("FP16 halves the size with no accuracy loss. Post-training INT8 is architecture-dependent: "
                   "the pretrained MobileNetV3 degrades badly (near-zero/outlier per-channel weight ranges) and would need "
                   "quantisation-aware training; WeedNet-Lite and ShuffleNetV2 lose nothing.")
    sel = st.selectbox("Show training details for", runs, index=runs.index(arch))
    a, b = st.columns(2)
    for col, f, cap in ((a, "curves.png", "Learning curves"), (b, "confusion_matrix.png", "Normalised confusion matrix")):
        p = os.path.join(RUNS, sel, f)
        if os.path.exists(p):
            col.image(p, caption=cap, width="stretch")
    rp = os.path.join(RUNS, sel, "test_report.txt")
    if os.path.exists(rp):
        st.code(open(rp).read(), language=None)

# ---------------------------------------------------------------- 5. about
with tabs[4]:
    st.markdown("""
### Pipeline
**DeepWeeds images → augmentation → lightweight CNN (transfer learning) → ONNX export → INT8 quantisation → real-time grid detection on an edge device**

| Step | What happens |
|---|---|
| Data | 17,509 field images, 8 weed species plus *Negative*. Stratified 70/15/15 split; class-balanced sampling because about 52% of images are Negative |
| Model | Depthwise-separable CNNs: MobileNetV3, ShuffleNetV2, EfficientNet-B0, and our custom **WeedNet-Lite** (0.47M params) |
| Training | AdamW, OneCycle learning rate, label smoothing 0.1, augmentation (crop, flip, rotate, colour jitter, erasing), early stopping |
| Edge optimisation | Export to ONNX → dynamic INT8 quantisation (about 4× smaller weights) → ONNX Runtime runs on Raspberry Pi, Jetson or phone |
| Detection | The frame is cut into an R×C grid, all tiles are classified in one batch, and tiles with weed confidence above the threshold are marked → weed map and coverage % |
| Explainability | Grad-CAM shows which pixels drove the decision |
| Rejecting non-field inputs | Deep k-nearest-neighbour OOD detection. The CNN feature vector is compared with 4,000 real training images, and the input is rejected if it is further away than 99% of the genuine validation images (adjustable in the sidebar). A trained 10th class, 'Not a field image', rejects people, rooms and objects as well. A photo of people or a room is therefore reported as "not a field image" instead of being forced into one of the 9 classes |
""")
