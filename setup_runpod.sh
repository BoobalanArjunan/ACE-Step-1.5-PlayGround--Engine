#!/bin/bash
echo "[1/4] Installing system dependencies (git-lfs)..."
apt-get update && apt-get install -y git-lfs

echo "[2/4] Downloading models via Git LFS..."
mkdir -p ./checkpoints
cd ./checkpoints
git lfs install
git clone https://huggingface.co/ACE-Step/Ace-Step1.5
mv Ace-Step1.5/* .
rm -rf Ace-Step1.5
cd ..

echo "[3/4] Installing UV package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "[4/4] Setting UV to safe-copy mode for RunPod NAS..."
export UV_LINK_MODE=copy

echo "Setup complete! You can now run ./train_parai_lora.sh"
