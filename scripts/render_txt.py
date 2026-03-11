"""
render_txt.py — Render text_gen.json + token_drift.json into a readable .txt report.

Format per domain:
  Code — def fibonacci(n): ... → completion:
  BF16          fib(n-1) + fib(n-2)                          0.000%
  Q8_0          fib(n-1) + fib(n-2)                          0.006%
  bart-Q6_K_L   fib(n-1) + fib(n-2)                          0.015%
  ...

Run: python scripts/render_txt.py
"""

import json
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
TEXT_GEN   = ROOT / "results" / "text_gen.json"
KLD_JSON   = ROOT / "results" / "token_drift.json"
OUTPUT_TXT = ROOT / "results" / "drift_report.txt"

# How many characters of completion to show per row
COMPLETION_WIDTH = 120


def load_json(p):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def compact(text, width):
    """Single-line, trimmed to width."""
    return text.replace("\n", " ").replace("\r", "").strip()[:width]


def main():
    gen = load_json(TEXT_GEN)
    kld = load_json(KLD_JSON)

    if not gen:
        print(f"ERROR: {TEXT_GEN} not found or empty. Run text_gen.py first.")
        return

    lines = []
    lines.append("QUANTIZATION DRIFT — TEXT GENERATION REPORT")
    lines.append("Qwen3.5-9B · llama.cpp · greedy (temp=0) · seed=42 · 100 tokens")
    lines.append("Mean KLD from token_drift.json (n_ctx=256, 127 scored tokens)")
    lines.append("=" * 100)

    for domain, completions in gen.items():
        if not completions:
            continue

        # Get KLD means for this domain
        kld_means = {}
        if domain in kld:
            for label, qdata in kld[domain].get("quants", {}).items():
                kld_means[label] = qdata.get("mean_kld", 0.0)

        # BF16 completion is the reference
        bf16_text = completions.get("BF16", "")

        # Build prompt display from domain name
        domain_prompts = {
            "Code":     "def fibonacci(n):\\n    if n <= 1: return n\\n    return",
            "Math":     "Solve x^2 + 5x + 6 = 0 step by step:",
            "Language": "A compass guides a traveler through unknown terrain, just as memory guides",
            "French":   "Le soleil se couchait lentement sur la ville, et Marie pensait à",
        }
        prompt_display = domain_prompts.get(domain, domain)

        lines.append("")
        lines.append(f"{'─'*100}")
        lines.append(f"{domain} — {prompt_display}")
        lines.append(f"{'─'*100}")

        label_w = 25

        for label, text in completions.items():
            mean_kld = kld_means.get(label, 0.0) if label != "BF16" else 0.0
            pct = f"{mean_kld * 100:.3f}%"

            completion = compact(text, COMPLETION_WIDTH)

            # Mark rows where text differs from BF16
            differs = (text.strip() != bf16_text.strip()) and label != "BF16"
            marker = " ◄" if differs else ""

            lines.append(f"  {label:<{label_w}}  {pct:>8}   {completion}{marker}")

        lines.append("")

    report = "\n".join(lines)
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
