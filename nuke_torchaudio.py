"""
nuke_torchaudio.py
==================
Runtime bypass for the PyTorch 2.10+ TorchCodec C++ crash on RunPod NAS environments.

Problem:
  - RunPod's /workspace is a NAS drive missing libavutil.so.60 (FFmpeg 8)
  - PyTorch 2.10 removed the backend="soundfile" fallback from torchaudio.load
  - This causes: "undefined symbol: torch_dtype_float4_e2m1fn_x2"

Solution:
  - This script patches the preprocess_audio.py inside the .venv at runtime
  - Replaces torchaudio.load with a pure soundfile implementation
  - Run ONCE before training, then run your training script normally

Usage:
  python nuke_torchaudio.py
  bash train_custom_lora.sh
"""

import os

VENV_PATCH_PATH = ".venv/lib/python3.11/site-packages/acestep/training/dataset_builder_modules/preprocess_audio.py"

PATCH_CODE = '''
import soundfile as sf
import torch
import torchaudio


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """
    Load audio using soundfile (bypasses broken torchaudio/torchcodec on RunPod NAS).
    Converts to stereo float32 tensor at the target sample rate.
    """
    audio_np, sr = sf.read(audio_path)
    audio = torch.from_numpy(audio_np).float()

    # Ensure shape is [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.t()

    # Resample if needed
    if sr != target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)

    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    # Trim to max duration
    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, target_sample_rate
'''

if os.path.exists(VENV_PATCH_PATH):
    with open(VENV_PATCH_PATH, "w") as f:
        f.write(PATCH_CODE)
    print(f"✅ Successfully patched: {VENV_PATCH_PATH}")
    print("   torchaudio.load replaced with soundfile bypass.")
    print("   You can now run your training script safely.")
else:
    print(f"❌ Could not find: {VENV_PATCH_PATH}")
    print("   Make sure you have run 'uv sync' at least once first.")
    print("   Also check your Python version (expects python3.11).")
