# 🥁 ACE-Step 1.5 — South Indian AI Music Playground

> **A fully working, production-ready local AI music generation engine with custom South Indian percussion LoRA fine-tuning — optimized for Apple Silicon.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black.svg?logo=apple)](https://www.apple.com/mac/)
[![Model: ACE-Step 1.5](https://img.shields.io/badge/Model-ACE--Step%201.5-orange.svg)](https://github.com/ace-step/ACE-Step)
[![LoRA on HuggingFace](https://img.shields.io/badge/🤗%20LoRA-ACE--Step--1.5--SouthIndiaParaiLoRA-yellow.svg)](https://huggingface.co/boobalanit/ACE-Step-1.5-SouthIndiaParaiLoRA)

---

## 🎵 What Is This?

This is a **community-enhanced fork** of the [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) music generation engine, extended with:

- ✅ **Apple Silicon (M1/M2/M3/M4) bug fixes** — bypasses the MLX precision corruption that causes static on large DiT models
- ✅ **A custom South Indian Parai percussion LoRA** — trained on 393 authentic studio-quality drum loops at 80–172 BPM
- ✅ **Cloud training infrastructure** — production RunPod scripts for fine-tuning your own cultural instrument LoRAs
- ✅ **A custom JSON metadata compiler** — for building structured multi-tag datasets from raw audio collections

---

## 🚨 Engineering Contributions

This repository documents and fixes **three critical bugs** in the original ACE-Step 1.5 engine:

### 1. 🍎 The Apple Silicon MLX Precision Bug
**Symptom:** The flagship `acestep-v15-xl-base` (4B parameter DiT) generated pure screeching static on all Apple Silicon Macs.

**Root Cause:** Apple's native `mlx` framework lacks the floating-point precision required for a 4-billion parameter Diffusion Transformer. As the audio propagated through 32 diffusion layers, numerical precision degraded progressively, corrupting the output tensor into noise. The smaller `acestep-turbo` model was unaffected due to its smaller parameter count.

**Fix:** Hard-coded `use_mlx_dit=False` in `acestep/ui/gradio/interfaces/generation_service_config_toggles.py`, forcing the engine onto PyTorch MPS (Apple Metal) backend which maintains correct precision for large models.

```python
# BEFORE (original — causes static on xl-base):
value=params.get("mlx_dit", mlx_ok) if service_pre_initialized else mlx_ok,

# AFTER (fixed — forces correct PyTorch MPS precision):
value=False,
```

---

### 2. 🐧 The RunPod NAS / PyTorch C++ Codec Bypass
**Symptom:** Audio preprocessing completely crashed on RunPod cloud GPUs with:
```
TorchCodec is required for load_with_torchcodec
undefined symbol: torch_dtype_float4_e2m1fn_x2
```

**Root Cause:** RunPod's `/workspace` is a Network Attached Storage (NAS) drive. PyTorch 2.10's `torchcodec` requires `libavutil.so.60` (FFmpeg 8 C++ library) which was missing from RunPod's Linux kernel image. PyTorch 2.10 had also fully deprecated the `backend="soundfile"` fallback parameter.

**Fix:** Dynamically replaced `torchaudio.load` inside the virtual environment at runtime with a pure `soundfile` implementation that bypasses the broken codec entirely:

```python
# nuke_torchaudio.py — Runtime C++ codec bypass
import soundfile as sf
import torch, torchaudio

def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    audio_np, sr = sf.read(audio_path)
    audio = torch.from_numpy(audio_np).float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.t()
    if sr != target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]
    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
    return audio, target_sample_rate
```

---

### 3. 📂 The RunPod NAS `uv` Hardlink Bug
**Symptom:** `uv sync` hung permanently at `141/142` packages during dependency installation on RunPod.

**Root Cause:** `uv` uses hardlinking by default for performance. Network Attached Storage (NAS) drives do not support hardlinks between different filesystem mount points. The process silently deadlocked trying to hardlink 3GB of PyTorch across filesystem boundaries.

**Fix:** `export UV_LINK_MODE=copy` forces `uv` to byte-copy files instead of hardlinking, which is fully compatible with NAS drives.

---

## 🚀 Quick Start (macOS Apple Silicon)

```bash
git clone https://github.com/YourUsername/ace-step-parai-playground
cd ace-step-parai-playground
./start_gradio_ui_macos.sh
```

The engine will automatically:
1. Install all Python dependencies via `uv`
2. Download the ACE-Step models from HuggingFace (~16GB)
3. Launch the Gradio UI at `http://127.0.0.1:7860`

> **Requires:** macOS 13+, Apple Silicon (M1/M2/M3/M4), 16GB+ RAM, 25GB free disk space

---

## 🎓 Training Your Own Cultural LoRA

This repository includes a complete pipeline to train your own instrument LoRA on RunPod cloud GPUs.

### Step 1: Prepare Your Dataset
Your audio files must be accompanied by a `.json` sidecar metadata file:
```json
{
  "caption": "South Indian parai festival drum loop, 120 BPM, 4/4 time",
  "bpm": 120,
  "keyscale": "C minor",
  "language": "instrumental",
  "audio_path": "your_drum_loop.wav"
}
```

### Step 2: Compile the Master Dataset
```bash
python compile_dataset_json.py --audio-dir ./your_audio_folder --output dataset.json
```

### Step 3: Run Cloud Training on RunPod
```bash
# Upload to RunPod and run:
bash setup_runpod.sh         # Fixes all RunPod NAS + codec bugs automatically
bash train_custom_lora.sh    # Trains your LoRA
```

> **Recommended:** RunPod A40 (48GB VRAM), 25–35 epochs for small datasets (< 500 files), learning rate `1e-4`

---

## 🥁 About the Parai

The South Indian Parai (பறை) is one of the world's oldest percussion instruments, central to Tamil folk, festival, and funeral traditions. This LoRA was trained on:

- **393 audio files** — authentic studio-quality Parai, Thappu, Thappattam, and Urumi drum loops
- **BPM range:** 80 – 172 BPM (covering slow ceremonial to fast Kuthu/Dappankuthu tempos)
- **Training:** 30 epochs on RunPod A40
- **Final loss:** 0.48 (75.5% reduction from baseline 1.97)

### Example Prompts

**Dappankuthu / Kuthu:**
```
High energy South Indian dappankuthu street festival drumming, massive thappu and parai
percussion ensemble, aggressive folk rhythm at 150 BPM, 4/4 time, deeply resonant bass
strikes, sharp metallic treble slaps, live and immersive stadium atmosphere
```

**Ceremonial / Slow:**
```
Traditional South Indian parai funeral procession drumming, slow and solemn 80 BPM,
deeply resonant bass, controlled thappu rhythm, temple procession atmosphere,
emotional and dignified, acoustic natural recording
```

---

## 🏗️ Repository Structure

```
ace-step-parai-playground/
├── acestep/                         # Core ACE-Step engine (with MLX bug fix)
├── train_custom_lora.sh             # LoRA training orchestrator
├── compile_dataset_json.py          # JSON metadata compiler
├── nuke_torchaudio.py               # RunPod C++ codec bypass
├── setup_runpod.sh                  # Automated RunPod environment setup
└── start_gradio_ui_macos.sh         # macOS launcher (MLX bypass applied)
```

---

## 👤 Author

**Boobalan Arjunan**  
GitHub: [@BoobalanArjunan](https://github.com/BoobalanArjunan)  
HuggingFace: [boobalanit](https://huggingface.co/boobalanit)

---

## 🙏 Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) — Original music generation engine
- [Hugging Face](https://huggingface.co/ACE-Step) — Model hosting

---

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE) for details.
