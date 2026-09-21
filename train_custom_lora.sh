#!/bin/bash

# ==============================================================
# train_custom_lora.sh
# ACE-Step 1.5 — Custom Cultural Instrument LoRA Training Script
# ==============================================================
#
# USAGE:
#   1. Set AUDIO_DIR to the folder containing your .wav + .json pairs
#   2. Set LORA_NAME to a name for your LoRA checkpoint
#   3. Run: bash train_custom_lora.sh
#
# REQUIREMENTS:
#   - RunPod A40 (48GB VRAM) or better recommended
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - Run nuke_torchaudio.py first if on RunPod NAS environment
#
# TIPS:
#   - For small datasets (< 500 files): use 25-35 epochs
#   - For large datasets (> 2000 files): use 10-20 epochs
#   - Learning rate 1e-4 is optimal for ACE-Step DiT models
# ==============================================================

# ── Configuration ──────────────────────────────────────────────
# CHANGE THESE to match your dataset and output paths:
AUDIO_DIR="/workspace/dataset/your_audio_folder"
LORA_NAME="MyCustomLoRA"

# ── Fixed paths (no need to change) ────────────────────────────
MASTER_JSON="${AUDIO_DIR}/master_dataset.json"
TENSOR_OUT="./training_data/${LORA_NAME}_tensors"
CHECKPOINT_OUT="./checkpoints/${LORA_NAME}"

# ── Fix RunPod NAS uv hardlink issue ───────────────────────────
export UV_LINK_MODE=copy

echo ""
echo "=================================================="
echo "  ACE-Step 1.5 Custom LoRA Training Pipeline"
echo "  LoRA Name : $LORA_NAME"
echo "  Audio Dir : $AUDIO_DIR"
echo "=================================================="
echo ""

# ── STEP 1: Compile JSON Sidecars ──────────────────────────────
echo "[ 1/3 ] Compiling sidecar .json files into master dataset..."
uv run python compile_dataset_json.py \
  --audio-dir "$AUDIO_DIR" \
  --output "$MASTER_JSON"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ STEP 1 FAILED: JSON compilation error."
    echo "   Check that your audio folder contains .json sidecar files."
    exit 1
fi
echo "✅ STEP 1 DONE: Master dataset compiled → $MASTER_JSON"
echo ""

# ── STEP 2: Preprocess Audio to Tensors ────────────────────────
echo "[ 2/3 ] Preprocessing audio to .pt tensors..."
uv run python train.py fixed \
  --preprocess \
  --audio-dir "$AUDIO_DIR" \
  --dataset-json "$MASTER_JSON" \
  --tensor-output "$TENSOR_OUT" \
  --checkpoint-dir ./checkpoints \
  --dataset-dir "$TENSOR_OUT" \
  --output-dir "$CHECKPOINT_OUT"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ STEP 2 FAILED: Audio preprocessing error."
    echo "   If you see a TorchCodec error, run: python nuke_torchaudio.py"
    exit 1
fi
echo "✅ STEP 2 DONE: Tensors saved → $TENSOR_OUT"
echo ""

# ── STEP 3: Fine-Tune the LoRA ─────────────────────────────────
echo "[ 3/3 ] Fine-tuning LoRA model (this will take a while)..."
uv run python train.py fixed \
  --checkpoint-dir ./checkpoints \
  --dataset-dir "$TENSOR_OUT" \
  --output-dir "$CHECKPOINT_OUT" \
  --epochs 30 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --adapter-type lora \
  --rank 64 \
  --alpha 128 \
  --cfg-ratio 0.15 \
  --optimizer-type adamw \
  --lr 1e-4

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training complete!"
    echo "   Your LoRA is saved to: $CHECKPOINT_OUT/final"
    echo ""
    echo "   To use it, copy the 'final' folder into:"
    echo "   ACE-Step-1.5/checkpoints/<YourLoRAName>/final"
    echo "   Then select it from the LoRA dropdown in the Gradio UI."
else
    echo ""
    echo "❌ STEP 3 FAILED: Training crashed."
    echo "   Check VRAM: this requires 40GB+ for batch_size=4."
    echo "   Try reducing --batch-size to 2 or --gradient-accumulation to 8."
    exit 1
fi
