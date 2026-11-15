#!/usr/bin/env bash
# End-to-end pipeline: data -> train -> evaluate -> export -> OOD calibration -> benchmark.
# v2: 10 classes (DeepWeeds + "Not a field image" outliers), longer training, MixUp/CutMix.
# The three models train in parallel; each logs to runs/<arch>.log.
# Usage: bash run_all.sh            (override epochs: EPOCHS_MBV3=20 bash run_all.sh)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
DATA=data/deepweeds

[ -f "$DATA/labels.csv" ] || $PY scripts/download_deepweeds.py "$DATA"
[ -f data/outliers/manifest.csv ] || $PY scripts/download_outliers.py
mkdir -p runs

run () {  # arch epochs extra-args...
  local arch=$1 epochs=$2; shift 2
  {
    echo "================= $arch ================="
    PYTHONUNBUFFERED=1 $PY train.py --data $DATA --arch "$arch" --epochs "$epochs" --workers 3 "$@"
    $PY evaluate.py --ckpt runs/$arch/best.pt --data $DATA --workers 3
    PYTHONUNBUFFERED=1 $PY export.py --ckpt runs/$arch/best.pt
    $PY calibrate_ood.py --run runs/$arch
    echo "DONE $arch"
  } > runs/$arch.log 2>&1
}

run mobilenet_v3_small "${EPOCHS_MBV3:-35}" &
run weednet_lite       "${EPOCHS_LITE:-50}" --no-pretrained --lr 3e-3 &
run shufflenet_v2      "${EPOCHS_SHUF:-30}" &
wait

$PY benchmark.py --save runs/benchmark_pytorch.json
$PY benchmark.py --save runs/benchmark_onnx.json --onnx runs/*/model.onnx runs/*/model.fp16.onnx runs/*/model.int8.onnx
$PY -c "import json; json.dump(json.load(open('runs/benchmark_pytorch.json'))+json.load(open('runs/benchmark_onnx.json')), open('runs/benchmark.json','w'), indent=2)"
echo "All done. Launch the UI with: .venv/bin/streamlit run app.py"
