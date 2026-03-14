"""
token_drift.py — Per-token KLD extractor for quantization drift visualization.

Generates one logits file per (domain, model), computes per-token KLD,
and outputs token_drift.json for the visualization.

Edit CONFIG below, then: python token_drift.py
Requires: numpy
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Project root = parent of the scripts/ folder
ROOT = Path(__file__).resolve().parent.parent

# Edit these to match your setup
import sys as _sys
LLAMA_PERPLEXITY = ROOT / "llama.cpp" / ("llama-perplexity.exe" if _sys.platform == "win32" else "llama-perplexity")

OUTPUT_DIR       = ROOT / "results"
LOGITS_DIR       = ROOT / "results" / "logits"

# vocab.json from the model repo — used to decode any token ID to text.
# Download from: https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/vocab.json
VOCAB_JSON       = ROOT / "models" / "baseline" / "vocab.json"

# n_ctx controls the window. Scored tokens = n_ctx // 2. Input must be >= 2*n_ctx tokens.
N_CTX = 256   # -> 128 scored tokens per domain (prompt needs >= 512 tokens)

# GPU layers
NGL = 99

# Models root — all .gguf files discovered automatically and sorted by size (BPW proxy).
MODELS_DIR = ROOT / "models"

# Subfolder -> short prefix used in labels
_PREFIX = {
    "baseline":             "",
    "bartowski":            "bart",
    "unsloth":              "unsl",
    "lmstudio-community":   "lmst",
}

def _build_models():
    """Discover all .gguf files, generate short labels, sort by file size descending."""
    entries = []  # (size, label, path)
    for gguf in sorted(MODELS_DIR.rglob("*.gguf")):
        # determine prefix from first-level subfolder name
        rel = gguf.relative_to(MODELS_DIR)
        top = rel.parts[0]
        prefix = _PREFIX.get(top, top[:4])

        # strip model-name prefix from filename to get the quant token
        stem = gguf.stem  # e.g. Qwen_Qwen3.5-9B-Q4_K_M
        for strip in ("Qwen_Qwen3.5-9B-", "Qwen3.5-9B-"):
            if stem.startswith(strip):
                stem = stem[len(strip):]
                break
        # normalise bf16 capitalisation
        if stem.lower() == "bf16":
            stem = "BF16"

        label = f"{prefix}-{stem}" if prefix else stem
        entries.append((gguf.stat().st_size, label, gguf))

    # sort largest (highest BPW) first, BF16 always first
    entries.sort(key=lambda e: (-e[0], e[1]))

    # preserve insertion order dict — BF16 must be first
    models = {}
    bf16 = [(l, p) for _, l, p in entries if l == "BF16"]
    rest  = [(l, p) for _, l, p in entries if l != "BF16"]
    for label, path in bf16 + rest:
        models[label] = path
    return models

MODELS = _build_models()

# Domain prompt files — must tokenize to >= 2*N_CTX tokens.
# We write them from PROMPTS below, or you can point to existing .txt files.
PROMPTS = {
    "Code": """\
# Classic algorithms in Python
# --------------------------------

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def binary_search(arr, target, lo=0, hi=None):
    if hi is None:
        hi = len(arr) - 1
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    if arr[mid] < target:
        return binary_search(arr, target, mid + 1, hi)
    return binary_search(arr, target, lo, mid - 1)

def power(base, exp):
    if exp == 0:
        return 1
    if exp % 2 == 0:
        half = power(base, exp // 2)
        return half * half
    return base * power(base, exp - 1)

def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def count_ways(n, coins):
    dp = [0] * (n + 1)
    dp[0] = 1
    for coin in coins:
        for amount in range(coin, n + 1):
            dp[amount] += dp[amount - coin]
    return dp[n]

def longest_common_subsequence(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def sum_digits(n):
    if n == 0:
        return 0
    return n % 10 + sum_digits(n // 10)

def reverse_string(s):
    if len(s) <= 1:
        return s
    return reverse_string(s[1:]) + s[0]

def depth(tree):
    if tree is None:
        return 0
    return 1 + max(depth(tree.get('left')), depth(tree.get('right')))

def combinations(lst, k):
    if k == 0:
        return [[]]
    if not lst:
        return []
    first = lst[0]
    rest = lst[1:]
    with_first = [[first] + c for c in combinations(rest, k - 1)]
    without_first = combinations(rest, k)
    return with_first + without_first
""",

    "Math": """\
Linear algebra and calculus review.

The derivative of f(x) = x^n is f'(x) = n * x^(n-1).
The integral of x^n dx = x^(n+1) / (n+1) + C, for n != -1.

Matrix multiplication: if A is m x n and B is n x p, then AB is m x p.
The determinant of a 2x2 matrix [[a, b], [c, d]] is ad - bc.
A matrix is invertible if and only if its determinant is nonzero.

Solving quadratic equations: ax^2 + bx + c = 0.
x = (-b +/- sqrt(b^2 - 4ac)) / (2a).
For x^2 + 5x + 6 = 0: x = (-5 +/- sqrt(25 - 24)) / 2 = (-5 +/- 1) / 2.
So x = -2 or x = -3. Check: (-2+2)(-2+3) = 0 and (-3+2)(-3+3) = 0. Correct.

The fundamental theorem of calculus states that if F'(x) = f(x), then
the integral from a to b of f(x) dx equals F(b) - F(a).

Euler's identity: e^(i*pi) + 1 = 0.
This connects five fundamental constants: e, i, pi, 1, and 0.

The Pythagorean theorem: a^2 + b^2 = c^2 for right triangles.
For a triangle with legs 3 and 4, the hypotenuse is sqrt(9 + 16) = sqrt(25) = 5.

Taylor series expansion of e^x around x=0:
e^x = 1 + x + x^2/2! + x^3/3! + x^4/4! + ...
This series converges for all real values of x.

Probability and combinatorics.
The number of ways to choose k items from n is C(n,k) = n! / (k! * (n-k)!).
C(5,2) = 5! / (2! * 3!) = 120 / 12 = 10.

Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B).
This is the foundation of Bayesian inference.

Linear systems can be solved with Gaussian elimination.
For 2x + y = 5 and x - y = 1: adding both equations gives 3x = 6, so x = 2, y = 1.

The eigenvalues of a matrix A satisfy det(A - lambda*I) = 0.
For A = [[2, 1], [0, 3]], det = (2-lambda)(3-lambda) = 0, so lambda = 2 or lambda = 3.

Fourier series represent periodic functions as sums of sines and cosines.
f(x) = a0 + sum(an * cos(n*x) + bn * sin(n*x)) for n = 1 to infinity.

Vector calculus: the gradient of f(x,y) = x^2 + y^2 is grad(f) = (2x, 2y).
The divergence of F = (P, Q) is dP/dx + dQ/dy.
The curl of a 3D vector field measures its rotation.

Complex numbers: z = a + bi where i^2 = -1.
The modulus is |z| = sqrt(a^2 + b^2).
Multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.

Statistics and probability distributions.
The mean of a set is the sum divided by the count.
The variance is the mean of squared deviations from the mean.
The standard deviation is the square root of the variance.
For a normal distribution, 68% of values lie within one standard deviation.
The binomial distribution models the number of successes in n independent trials.
P(X=k) = C(n,k) * p^k * (1-p)^(n-k).
The Poisson distribution models rare events: P(X=k) = e^(-lambda) * lambda^k / k!.
Linear regression finds the line y = mx + b minimizing squared residuals.
""",

    "Language": """\
Memory, identity, and the passage of time.

A compass guides a traveler through unknown terrain, pointing always toward
the magnetic north. In the same way, memory guides a person through the
uncharted landscape of their own past, connecting who they were to who they
are. Without memory, identity dissolves like morning fog under the summer sun.

The philosopher Henri Bergson argued that memory is not a drawer in which we
store fixed images of the past. It is a living process, constantly reshaping
itself in light of the present. Each time we remember something, we subtly
alter the memory itself. The act of remembering is also an act of creating.

Language is the vessel in which memories travel through time. A word can
carry an entire civilization forward across centuries. The word fire meant
warmth and danger to our ancestors, and it means exactly the same thing today.
Some meanings are so fundamental that no amount of historical drift can
erode them entirely.

Time, however, is not neutral. It compresses certain events and stretches
others. A childhood summer can feel longer in memory than an entire decade
of adult routine. The intensity of experience, not its objective duration,
determines how much space it occupies in the mind.

To remember well is a form of wisdom. To forget gracefully is another.

Consciousness and perception.

What we call reality is a construction. The brain receives incomplete signals
from the senses and fills in the rest from expectation and prior experience.
We do not see the world as it is. We see the world as we are.

Every perception is a hypothesis. When you recognize a face across a crowded
room, your brain has run a probabilistic match against millions of stored
patterns and returned its best guess. The certainty you feel is not evidence
of accuracy. It is evidence of commitment.

Attention is the mechanism by which the brain decides what matters. In any
given moment, an enormous quantity of information competes for processing.
Most of it is discarded before it reaches awareness. What survives is not
necessarily what is most true, but what is most relevant to current goals.

Sleep consolidates memory by replaying the day's experience in compressed
form. The hippocampus transfers patterns to the neocortex during slow-wave
sleep. Dreams may be a side effect of this process, or they may serve a
regulatory function we do not yet fully understand.

Language shapes thought in ways that are difficult to see from inside a
single language. The boundaries of what can be easily said influence the
boundaries of what can be easily thought. This is not determinism but
it is a real and measurable effect.

The nature of knowledge and belief.

Knowledge is not merely true belief. A clock that has stopped is right
twice a day, but we would not say it knows the time. Something more is
required — a reliable connection between the belief and the fact. That
extra ingredient is what epistemologists have spent centuries trying to name.

Skepticism is the position that we cannot know anything with certainty.
Descartes demonstrated that even the most obvious beliefs — that I have
hands, that the sun will rise tomorrow — rest on assumptions that cannot
be fully verified from the inside. His response was the cogito: the act
of doubting proves that the doubter exists.

Pragmatism takes a different approach. A belief is true if it works, if
it helps us navigate the world successfully. This shifts the question
from correspondence with reality to practical effectiveness.
""",

    "French": """\
La memoire et le temps qui passe.

Le soleil se couchait lentement sur la ville, teintant le ciel de nuances
d'orange et de rose. Marie regardait par la fenetre et pensait a son enfance,
aux etes dores ou le temps semblait s'arreter pour elle seule. Elle se
souvenait des jardins de sa grand-mere, du parfum des roses et de la lavande,
du bruit des cigales dans la chaleur de l'apres-midi.

La memoire est une chose etrange. Elle preserve certains moments avec une
precision remarquable, tandis qu'elle laisse d'autres s'effacer comme de
l'encre sous la pluie. On ne choisit pas ce qu'on retient. C'est la memoire
qui choisit pour nous, selon des criteres mysterieux qui echappent a la raison.

Marcel Proust avait compris cette verite mieux que quiconque. Dans son grand
roman, une simple madeleine trempee dans du the suffit a faire surgir tout un
monde englouti. Les sensations sont les gardiens les plus fideles du passe.
Elles resistent la ou les mots et les images s'effacent.

Le temps passe, mais certaines choses demeurent. Une melodie entendue dans
l'enfance peut traverser cinquante ans et resurgir intacte, aussi fraiche
qu'au premier jour. C'est le paradoxe de la memoire : elle est a la fois
fragile et indestructible, selective et totale.

Nous sommes tous faits de nos souvenirs. Sans eux, nous ne serions que
des etrangers a nous-memes, errant sans boussole dans le present.

La langue et la pensee.

On ne pense pas de la meme facon dans toutes les langues. Chaque langue
decoupe le monde a sa maniere, impose ses categories, ses distinctions, ses
silences. Apprendre une nouvelle langue, c'est apprendre a voir autrement.

Le français a cette particularite de distinguer le tutoiement du vouvoiement.
Cette distinction n'est pas seulement grammaticale : elle est sociale, affective,
politique. Elle encode dans la langue elle-meme toute une cartographie des
relations humaines.

Les mots que nous n'avons pas sont aussi importants que ceux que nous avons.
L'absence d'un mot ne signifie pas l'absence de la chose, mais elle rend
cette chose plus difficile a saisir, a communiquer, a partager.

La traduction est une forme de trahison necessaire. Toute traduction est
une interpretation. Elle revele autant sur le traducteur que sur l'auteur
original. Les grandes oeuvres resistent a la traduction parce qu'elles sont
faites de la texture meme de leur langue, de ses rythmes, de ses ambiguites.

Ecrire, c'est choisir. Chaque mot exclut tous les autres mots possibles.
Chaque phrase ferme des portes en meme temps qu'elle en ouvre. Le style,
c'est la somme de tous ces choix, conscients ou non.

La science et la connaissance.

La methode scientifique repose sur l'observation, l'hypothese et la
verification experimentale. Une theorie scientifique n'est jamais
definitivement prouvee : elle est seulement non refutee jusqu'a present.
C'est le principe de refutabilite de Popper.

La physique quantique a bouleverse notre conception de la realite. Les
particules elementaires n'ont pas de position definie avant d'etre
observees. L'incertitude n'est pas un defaut de nos instruments : elle
est constitutive de la nature elle-meme.

La relativite d'Einstein a montre que le temps et l'espace ne sont pas
absolus. Le temps s'ecoule plus lentement pres d'une masse importante.
Deux observateurs en mouvement relatif ne s'accordent pas sur la
simultaneite de deux evenements. Ces idees, contre-intuitives, sont
verifiees experimentalement avec une precision remarquable.
""",
}

# ---------------------------------------------------------------------------
# METHODOLOGY NOTE — DRIFT THRESHOLD
# ---------------------------------------------------------------------------
# The drift marker (│) in the visualization marks the first token where
# KLD(BF16 || quant) exceeds this threshold.
#
# We use 0.01 nats (fixed absolute threshold) for the following reasons:
#
#   1. It is model-independent and dataset-independent — anyone can reproduce
#      the marker by applying the same threshold to the same logits files.
#
#   2. At KLD = 0.01 nats, the expected log-probability difference between
#      the BF16 and quantized distributions is already perceptible at the
#      token level. Q8_0 never crosses it (mean 0.0001–0.001). Q2_K crosses
#      it consistently within the first few tokens.
#
#   3. It corresponds to a well-established reference point in the
#      information-theoretic literature for "noticeable" distribution shift.
#      (cf. van Erven & Harremos, 2014, IEEE Trans. Inf. Theory)
#
#   4. It requires no per-quant normalization or relative scaling, which
#      would introduce arbitrary choices that are harder to justify.
#
# To reproduce: run llama-perplexity with --kl-divergence-base on BF16 to
# get the reference logits, then again on any quant with --kl-divergence.
# Decode the uint16 log-prob format (see read_logits()), compute KLD per
# token as sum(P * (log P - log Q)), and apply threshold = 0.01.
# ---------------------------------------------------------------------------
DRIFT_THRESHOLD = 0.01

# ---------------------------------------------------------------------------
# Binary reader (llama.cpp --kl-divergence-base format)
# Header: "_logits_" (8) | n_ctx int32 | n_vocab int32 | n_chunk int32
#         | tokens int32[n_chunk*n_ctx]
# Data per token: scale f32 | min_log_prob f32 | uint16[n_vocab]
# nv = 2*((n_vocab+1)//2) + 4  uint16s per token
# ---------------------------------------------------------------------------

def read_logits(path: Path):
    with open(path, "rb") as f:
        magic   = f.read(8)
        assert magic == b"_logits_", f"bad magic: {magic}"
        n_ctx   = struct.unpack("<i", f.read(4))[0]
        n_vocab = struct.unpack("<i", f.read(4))[0]
        n_chunk = struct.unpack("<i", f.read(4))[0]
        f.read(n_chunk * n_ctx * 4)          # skip token ids
        nv  = 2 * ((n_vocab + 1) // 2) + 4  # uint16s per token row
        raw = np.frombuffer(f.read(), dtype=np.uint16)
    n_tok = len(raw) // nv
    data  = raw[: n_tok * nv].reshape(n_tok, nv)
    return data, n_vocab, nv


def decode_log_probs(row, n_vocab):
    """Decode one compressed token row into log-prob array."""
    scale  = np.frombuffer(row[:2].tobytes(), dtype=np.float32)[0]
    min_lp = np.frombuffer(row[2:4].tobytes(), dtype=np.float32)[0]
    return scale * row[4: 4 + n_vocab].astype(np.float32) + min_lp


def kld_per_token(p_data, q_data, n_vocab):
    """
    KLD(P||Q) for each scored token position.
    Returns: klds float32[n], p_tops int32[n], q_tops int32[n]
    """
    n      = min(len(p_data), len(q_data))
    klds   = np.zeros(n, dtype=np.float32)
    p_tops = np.zeros(n, dtype=np.int32)
    q_tops = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p_lp = decode_log_probs(p_data[i], n_vocab)
        q_lp = decode_log_probs(q_data[i], n_vocab)
        p_lp -= np.logaddexp.reduce(p_lp)
        q_lp -= np.logaddexp.reduce(q_lp)
        klds[i]   = float(np.sum(np.exp(p_lp) * (p_lp - q_lp)))
        p_tops[i] = int(np.argmax(p_lp))
        q_tops[i] = int(np.argmax(q_lp))
    return klds, p_tops, q_tops


# ---------------------------------------------------------------------------
# llama-perplexity runner
# ---------------------------------------------------------------------------

def load_vocab(path: Path) -> dict:
    """
    Load vocab.json (text -> id) and invert to {id -> text}.
    Covers every possible token ID including quant substitutes.
    """
    with open(path, "r", encoding="utf-8") as f:
        text_to_id = json.load(f)
    return {v: k for k, v in text_to_id.items()}


def read_scored_token_ids(path: Path) -> list:
    """
    Read the scored token IDs from the logits file header.
    Scored tokens are the second half of the context window: [n_ctx//2 : n_ctx].
    These are the tokens whose KLD values we computed.
    """
    with open(path, "rb") as f:
        f.read(8)                                        # magic
        n_ctx   = struct.unpack("<i", f.read(4))[0]
        f.read(4)                                        # n_vocab
        n_chunk = struct.unpack("<i", f.read(4))[0]
        token_ids = struct.unpack(
            f"<{n_chunk * n_ctx}i", f.read(n_chunk * n_ctx * 4)
        )
    first = n_ctx // 2
    # scored tokens are [first..n_ctx-1] — these are the tokens being predicted,
    # so the TEXT shown is token[first+1..n_ctx] (the next token after each position)
    return list(token_ids[first + 1 : n_ctx + 1])


def run_perplexity(model: Path, prompt_file: Path, logits_out: Path, is_bf16: bool = False) -> bool:
    if logits_out.exists():
        print(f"  [reuse] {logits_out.name}")
        return True
    cmd = [
        str(LLAMA_PERPLEXITY),
        "-m", str(model),
        "-f", str(prompt_file),
        "-c", str(N_CTX),
        "-ngl", str(NGL),
        "--chunks", "1",
        "--kl-divergence-base", str(logits_out),
    ]
    print(f"  [run]   {model.name}")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    # Minimum valid file: magic(8) + n_ctx(4) + n_vocab(4) + n_chunk(4) + at least 1 token id(4) = 24 bytes
    if r.returncode != 0 or not logits_out.exists() or logits_out.stat().st_size < 24:
        print(f"  [ERROR] returncode={r.returncode} size={logits_out.stat().st_size if logits_out.exists() else 0}")
        print(f"  [ERROR] prompt may be too short for N_CTX={N_CTX} (need >= {N_CTX} tokens)")
        print(f"  [STDERR] {r.stderr[-300:]}")
        # bad file left on disk intentionally — inspect to diagnose
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)

    prompt_dir = OUTPUT_DIR / "prompts"
    prompt_dir.mkdir(exist_ok=True)

    # Write prompt files
    prompt_files = {}
    for domain, text in PROMPTS.items():
        p = prompt_dir / f"{domain}.txt"
        p.write_text(text, encoding="utf-8")
        prompt_files[domain] = p

    model_labels = list(MODELS.keys())
    bf16_label   = model_labels[0]

    # Load vocab once — covers all token IDs across all domains and quants
    id_to_text = load_vocab(VOCAB_JSON)
    print(f"Loaded vocab: {len(id_to_text)} tokens")

    all_results = {}

    for domain, pfile in prompt_files.items():
        print(f"\n{'='*60}\nDomain: {domain}\n{'='*60}")

        # --- Generate logits for all models ---
        logit_files = {}
        for label, mpath in MODELS.items():
            if not mpath.exists():
                print(f"  [missing] {label}")
                continue
            lf = LOGITS_DIR / f"{domain}_{label}.bin"
            ok = run_perplexity(mpath, pfile, lf)
            if ok:
                logit_files[label] = lf

        if bf16_label not in logit_files:
            print(f"  [skip] BF16 logits missing for {domain}")
            continue

        # --- Scored token IDs from logits header ---
        scored_ids = read_scored_token_ids(logit_files[bf16_label])

        # --- Load BF16 reference ---
        bf16_data, n_vocab, nv = read_logits(logit_files[bf16_label])
        n_tok = len(bf16_data)
        print(f"  scored tokens: {n_tok}  n_vocab: {n_vocab}")

        # Map each scored position to its token text
        token_texts = [
            id_to_text.get(tid, f"<{tid}>") for tid in scored_ids[:n_tok]
        ]

        domain_out = {
            "n_tokens":    n_tok,
            "token_texts": token_texts,
            "quants":      {}
        }

        # --- Compare each quant to BF16 ---
        for label in model_labels[1:]:
            if label not in logit_files:
                continue
            q_data, _, _ = read_logits(logit_files[label])
            klds, p_tops, q_tops = kld_per_token(bf16_data, q_data, n_vocab)

            drift_idx  = next((i for i, v in enumerate(klds) if v > DRIFT_THRESHOLD), -1)
            alt_tokens = {
                str(i): {
                    "bf16":       int(p_tops[i]),
                    "bf16_text":  id_to_text.get(int(p_tops[i]), f"<{p_tops[i]}>"),
                    "quant":      int(q_tops[i]),
                    "quant_text": id_to_text.get(int(q_tops[i]), f"<{q_tops[i]}>"),
                }
                for i in range(len(klds)) if p_tops[i] != q_tops[i]
            }

            domain_out["quants"][label] = {
                "kld":        [round(float(v), 6) for v in klds],
                "mean_kld":   round(float(klds.mean()), 6),
                "drift_idx":  drift_idx,
                "alt_tokens": alt_tokens,
            }
            same = sum(p_tops[i] == q_tops[i] for i in range(len(klds)))
            print(f"  {label:20s}  mean_kld={klds.mean():.5f}  "
                  f"drift@{drift_idx:3d}  same_top={same}/{len(klds)}")

        all_results[domain] = domain_out

    # --- Save JSON ---
    out = OUTPUT_DIR / "token_drift.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
