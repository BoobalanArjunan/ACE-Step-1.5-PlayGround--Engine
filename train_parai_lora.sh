#!/bin/bash

# ==================================================
# South India Parai LoRA - Automated Training Script
# ==================================================

# Define your dataset paths here. 
# (Update these paths to your RunPod Linux paths before running on RunPod!)
AUDIO_DIR="/Users/mymac/Desktop/ACE Tranning Data/Lite Version/EM_INDIAN STREET DRUMMERS_Lite/EM_INDIAN STREET DRUMMERS_120BPM"
MASTER_JSON="${AUDIO_DIR}/master_dataset.json"

# Output directories
TENSOR_OUT="./training_data/parai_tensors"
CHECKPOINT_OUT="./checkpoints/SouthIndiaParaiLoRA"

echo "=================================================="
echo " STEP 1: Compiling Sidecar JSONs into Master JSON"
echo "=================================================="

# Run the python script to automatically gather all your individual .json files 
# into one master list that the AI trainer can understand.
uv run python compile_dataset_json.py --audio-dir "$AUDIO_DIR" --output "$MASTER_JSON"

if [ $? -ne 0 ]; then
    echo "❌ JSON Compilation failed! Please check the compile_dataset_json.py script."
    exit 1
fi

echo "=================================================="
echo " STEP 2: Preprocessing Audio to Tensors"
echo "=================================================="

# The --preprocess flag converts the .wav files to .pt tensors using the master JSON
uv run python train.py fixed \
  --preprocess \
  --audio-dir "$AUDIO_DIR" \
  --dataset-json "$MASTER_JSON" \
  --tensor-output "$TENSOR_OUT" \
  --checkpoint-dir ./checkpoints \
  --dataset-dir "$TENSOR_OUT" \
  --output-dir "$CHECKPOINT_OUT"

if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed! Please check your audio and JSON paths."
    exit 1
fi

echo "=================================================="
echo " STEP 3: Fine-Tuning the LoRA Model"
echo "=================================================="

# Train the model (BATCH SIZE IS 4 FOR RUNPOD A40!)
uv run python train.py fixed \
  --checkpoint-dir ./checkpoints \
  --dataset-dir "$TENSOR_OUT" \
  --output-dir "$CHECKPOINT_OUT" \
  --epochs 100 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --adapter-type lora \
  --rank 64 \
  --alpha 128 \
  --cfg-ratio 0.15 \
  --optimizer-type adamw \
  --lr 1e-4

if [ $? -eq 0 ]; then
    echo "✅ Training complete! Your Parai LoRA is saved to: $CHECKPOINT_OUT"
else
    echo "❌ Training failed during execution."
fi
