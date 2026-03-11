# Quantization Drift — Qwen3.5-9B

**Token-level drift across 47 quants of Qwen3.5-9B.**

**[Live visualization](https://huggingface.co/spaces/cmh/Qwen3.5-9B-GGUF-quant-drift)**

![outputs/preview.png](https://github.com/cmhamiche/token_drift/blob/main/preview.png)

---

## What's in here

```
scripts/
  text_gen.py          Generate completions from all 47 quants via llama-server router mode
  token_drift.py       Compute per-token KLD against BF16 using llama-perplexity
  render_txt.py        Combine both results into a readable drift_report.txt
  models-preset.ini    llama-server router config (edit paths before use)

results/
  text_gen.json        Raw completions for all 47 models × 4 domains
  token_drift.json     Per-token KLD arrays and drift indices (8-model subset)

outputs/
  quant_drift.html     Standalone interactive visualization (no build needed)
```

---

## Method

### Text completion divergence (`text_gen.py`)

- Starts `llama-server` in router mode with `--models-preset` and `--no-models-autoload`
- Loads each model into VRAM one at a time, runs greedy decode (temp=0, seed=42, 400 tokens)
- Records the first character position where each model's output diverges from BF16
- 4 domains: Code, Math, English, French

### Per-token KL divergence (`token_drift.py`)

- Uses `llama-perplexity --kl-divergence-base` to dump full logit distributions
- Computes KLD(BF16 ∥ quant) per scored token position
- Drift marker = first token where KLD > 0.01 nats
- 8-model subset (BF16, Q8_0, Q6_K_L, Q4_K_M ×3, Q3_K_M, Q2_K)

---

## Setup

### Requirements

- Python 3.8+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — needs both `llama-server` and `llama-perplexity`
- CUDA GPU recommended (scripts default to `-ngl 99`)
- `numpy` for `token_drift.py` only: `pip install numpy`

### Model files

Download Qwen3.5-9B GGUFs from HuggingFace and place them under `models/`:

```
models/
  baseline/
    Qwen3.5-9B-bf16.gguf
    Qwen3.5-9B-Q8_0.gguf
    vocab.json                  ← from https://hf.co/Qwen/Qwen3.5-9B/raw/main/vocab.json
  bartowski/Qwen_Qwen3.5-9B-GGUF/
    Qwen_Qwen3.5-9B-Q6_K_L.gguf
    ... (see models-preset.ini for full list)
  unsloth/Qwen3.5-9B-GGUF/
    ...
  lmstudio-community/Qwen3.5-9B-GGUF/
    ...
```

Place the `llama.cpp` binaries at `llama.cpp/llama-server[.exe]` and `llama.cpp/llama-perplexity[.exe]`, or edit the `LLAMA_SERVER` / `LLAMA_PERPLEXITY` paths at the top of each script.

Edit `scripts/models-preset.ini` to point `model =` lines at your actual `.gguf` paths.

### Run

```bash
# Step 1 — generate completions (≈2–4 hrs for 47 models on RTX 3060)
python scripts/text_gen.py

# Step 2 — compute per-token KLD (optional, 8-model subset)
python scripts/token_drift.py

# Step 3 — render a plain-text report combining both
python scripts/render_txt.py
```

Results land in `results/text_gen.json`, `results/token_drift.json`, and `results/drift_report.txt`.

---

## Hardware & software

| | |
|---|---|
| GPU | NVIDIA RTX 3060 12 GB |
| OS | Windows 11 26100.7840 |
| llama.cpp | 8250 (0beb8db3a) |
| CUDA drivers | 591.85 |
| Decode | temp=0, seed=42 |

---

## Models tested

47 GGUFs from three quantizers:
- **bartowski** — 23 variants (Q6_K_L → IQ2_S)
- **unsloth** — 20 variants including UD (Unsloth Dynamic) series
- **lmstudio-community** — 2 variants (Q6_K, Q4_K_M)
- **baseline** — BF16 + Q8_0

See `scripts/models-preset.ini` for the complete list with download paths.

---

## License

Scripts: MIT. Model weights are subject to their respective HuggingFace repository licenses.
