"""Exp G: extrapolate the measured SSD-tier ratios to large models on a Mac Studio.

MEASURED here (results/combined_*.json, results/compress.json):
  append-only vs full rewrite: exact arithmetic of the workload (no assumption)
  lossless split-LZFSE ratio: 1.18x fp16 KV, 1.34x bf16 KV
  8-bit storage size: 0.531 of raw (incl. scales/biases); 4/4 greedy outputs unchanged
ASSUMED (community / review numbers, see README sources):
  SSD sequential read: M3 Ultra ~5 GB/s, M5-generation Studio ~13 GB/s (2.6x M4 Max)
  prefill at long context, 4-bit: ranges below
"""
KB = 1024
MODELS = {  # bytes of KV per token (16-bit), KV dtype, prefill tok/s range at 32K (assumed)
    "Llama-3.2-3B (measured on M4 laptop)": (112 * KB, "fp16", (290, 330)),
    "Llama-3.3-70B dense": (80 * 8 * 128 * 2 * 2, "bf16", (100, 250)),
    "Qwen3-235B-A22B MoE": (94 * 4 * 128 * 2 * 2, "bf16", (250, 500)),
    "DeepSeek-V3/R1 671B (MLA)": (61 * (512 + 64) * 2, "bf16", (150, 210)),
}
CTX, TURN = 32768, 1024           # agent session grows to 32K, saved after every 1K-token turn
SAVES = CTX // TURN
LZ = {"fp16": 1.18, "bf16": 1.34}
Q8 = 0.531
SSD = {"M3 Ultra": 5e9, "M5-gen Studio": 13e9}
SESSIONS_PER_DAY = 20

print(f"Agent session: {CTX} tokens, saved every {TURN} tokens ({SAVES} saves); {SESSIONS_PER_DAY} sessions/day\n")
print("| Model | KV/token | KV at 32K | Recompute 32K | Cold resume raw (M3U / M5) | Cold resume 8-bit (M3U / M5) "
      "| SSD writes/session: full rewrite | append | append+lossless | append+8-bit | Writes/year (full → append+8-bit) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, (b, dt, (lo, hi)) in MODELS.items():
    kv = CTX * b
    full = sum(i * TURN for i in range(1, SAVES + 1)) * b
    app = CTX * b
    gb = lambda x: f"{x / 1e9:.1f} GB"
    rr = " / ".join(f"{kv / s:.1f} s" for s in SSD.values())
    rq = " / ".join(f"{kv * Q8 / s:.1f} s" for s in SSD.values())
    yr = lambda x: f"{x * SESSIONS_PER_DAY * 365 / 1e12:.0f} TB"
    print(f"| {name} | {b / KB:.0f} KB | {gb(kv)} | {CTX / hi:.0f}–{CTX / lo:.0f} s | {rr} | {rq} | {gb(full)} | {gb(app)} "
          f"| {gb(app / LZ[dt])} | {gb(app * Q8)} | {yr(full)} → {yr(app * Q8)} |")
