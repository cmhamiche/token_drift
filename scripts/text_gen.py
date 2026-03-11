"""
text_gen.py — Generate text completions from all quant models via llama-server router mode.
Uses models-preset.ini so each model is addressable by label name.
Run: python scripts/text_gen.py
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Project root = parent of the scripts/ folder
ROOT = Path(__file__).resolve().parent.parent

# Edit these to match your setup
LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server.exe"   # Windows
# LLAMA_SERVER = ROOT / "llama.cpp" / "llama-server"      # Linux / macOS

PRESET_FILE  = ROOT / "scripts" / "models-preset.ini"
OUTPUT_DIR   = ROOT / "results"

SERVER_HOST  = "127.0.0.1"
SERVER_PORT  = 8091
SERVER_URL   = f"http://{SERVER_HOST}:{SERVER_PORT}"

N_TOKENS     = 400
N_SEED       = 42
TEMP         = 0.0

# Model labels in BPW order (highest to lowest) — must match [sections] in preset
MODELS = [
    "BF16",
    "unsl-UD-Q8_K_XL",
    "Q8_0",
    "bart-Q6_K_L",
    "bart-Q6_K",
    "unsl-Q6_K",
    "lmst-Q6_K",
    "bart-Q5_K_L",
    "unsl-UD-Q5_K_XL",
    "bart-Q5_K_M",
    "unsl-Q5_K_M",
    "bart-Q5_K_S",
    "unsl-Q5_K_S",
    "bart-Q4_K_L",
    "unsl-UD-Q4_K_XL",
    "bart-Q4_K_M",
    "unsl-Q4_K_M",
    "lmst-Q4_K_M",
    "bart-Q4_K_S",
    "unsl-Q4_K_S",
    "bart-Q4_1",
    "unsl-Q4_1",
    "bart-Q4_0",
    "unsl-Q4_0",
    "bart-IQ4_NL",
    "unsl-IQ4_NL",
    "bart-IQ4_XS",
    "unsl-IQ4_XS",
    "bart-Q3_K_XL",
    "unsl-UD-Q3_K_XL",
    "bart-Q3_K_L",
    "bart-Q3_K_M",
    "unsl-Q3_K_M",
    "bart-Q3_K_S",
    "unsl-Q3_K_S",
    "bart-IQ3_M",
    "bart-IQ3_XS",
    "bart-IQ3_XXS",
    "unsl-UD-IQ3_XXS",
    "bart-Q2_K_L",
    "bart-Q2_K",
    "unsl-UD-Q2_K_XL",
    "bart-IQ2_M",
    "unsl-UD-IQ2_M",
    "bart-IQ2_S",
    "unsl-UD-IQ2_XXS",
]

PROMPTS = {
    "Code":     "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Math":     "Solve x^2 + 5x + 6 = 0 step by step:\n\nStep 1:",
    "Language": "A compass guides a traveler through unknown terrain, just as memory guides",
    "French":   "Le soleil se couchait lentement sur la ville, et Marie pensait à",
}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def start_server() -> subprocess.Popen:
    cmd = [
        str(LLAMA_SERVER),
        "--models-preset", str(PRESET_FILE),
        "--models-max", "1",
        "--no-models-autoload",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--log-disable",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_server(timeout=60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as r:
                status = json.loads(r.read()).get("status", "")
                if status in ("ok", "no slot available"):
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def load_model(label: str) -> bool:
    payload = json.dumps({"model": label}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/models/load",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
            return data.get("success", False)
    except Exception as e:
        print(f"load error: {e}")
        return False


def wait_for_model_loaded(label: str, timeout=180) -> bool:
    """Poll /models until the model status is 'loaded'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/models", timeout=5) as r:
                data = json.loads(r.read())
                for m in data.get("data", []):
                    if m.get("id") == label:
                        status = m.get("status", {}).get("value", "")
                        if status == "loaded":
                            return True
                        if status == "failed":
                            return False
        except Exception:
            pass
        time.sleep(2)
    return False


def unload_model(label: str):
    payload = json.dumps({"model": label}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/models/unload",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

def complete(prompt: str, model: str) -> str | None:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "n_predict": N_TOKENS,
        "seed": N_SEED,
        "temperature": TEMP,
        "top_k": 1,
        "top_p": 1.0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
            text = data.get("content", "").strip()
            return text or None
    except Exception as e:
        print(f"completion error: {e}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "text_gen.json"

    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Resuming — {sum(len(v) for v in results.values())} completions already saved.")
    else:
        results = {domain: {} for domain in PROMPTS}

    print("Starting llama-server (router mode)...")
    proc = start_server()
    if not wait_for_server():
        print("ERROR: server did not start")
        proc.kill()
        return
    print("Server ready.\n")

    total = len(MODELS)

    try:
        for idx, label in enumerate(MODELS, 1):
            already_done = all(label in results.get(d, {}) for d in PROMPTS)
            if already_done:
                print(f"[{idx}/{total}] [skip]  {label}")
                continue

            print(f"[{idx}/{total}] [load]  {label} ... ", end="", flush=True)
            if not load_model(label):
                print("LOAD FAILED")
                continue
            if not wait_for_model_loaded(label):
                print("LOAD TIMEOUT")
                continue
            print("ok")

            for domain, prompt in PROMPTS.items():
                if label in results.get(domain, {}):
                    print(f"         [skip]  {domain}")
                    continue

                print(f"         [gen]   {domain} ... ", end="", flush=True)
                text = complete(prompt, label)

                if text is None:
                    print("FAILED")
                    continue

                preview = text[:80].replace("\n", "↵")
                print(preview)

                results.setdefault(domain, {})[label] = text
                out_path.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            unload_model(label)
            time.sleep(1)

    finally:
        proc.terminate()
        proc.wait()

    print(f"\nDone. Saved: {out_path}")

    print("\n── Preview " + "─"*50)
    for domain, completions in results.items():
        bf16 = completions.get("BF16", "")
        print(f"\n{domain}:")
        for label, text in list(completions.items())[:6]:
            preview = text[:90].replace("\n", "↵")
            mark = " ◄" if text != bf16 and label != "BF16" else ""
            print(f"  {label:25s}  {preview}{mark}")


if __name__ == "__main__":
    main()
