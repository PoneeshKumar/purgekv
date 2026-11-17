"""Estimate (not measure) streamed resume for chat sizes, models and Macs you can't run.

  python research/scale_estimate.py [--fit results/scale_X.json] [--ram 64] [--ssd 6] [--new 256]

Model:  load    = kv_bytes / ssd_bw
        compute = new * (a + b * hist)          (per-token MLP cost + attention over history)
        streamed TTFT ~ max(load, compute) + one layer of the smaller, load+compute without streaming
--fit takes a bench_scale.py result: ssd_bw, a and b are fitted to its measurements,
and the fitted model is printed next to the measured numbers so you can judge it.
Without --fit it uses the M4 MacBook values below. Outputs are estimates.
"""
import argparse, json
import numpy as np

MODELS = {  # KV bytes/token (16-bit), layers, 4-bit weights GB  (published configs)
    "Llama-3.2-1B": (16 * 8 * 64 * 2 * 2, 16, 0.7),
    "Llama-3.2-3B": (28 * 8 * 128 * 2 * 2, 28, 1.8),
    "Llama-3.1-8B": (32 * 8 * 128 * 2 * 2, 32, 4.5),
    "Qwen3-14B": (40 * 8 * 128 * 2 * 2, 40, 8.3),
}
ap = argparse.ArgumentParser()
ap.add_argument("--fit")
ap.add_argument("--ram", type=float, default=16)
ap.add_argument("--ssd", type=float, help="SSD GB/s (overrides the fit)")
ap.add_argument("--new", type=int, default=32)
ap.add_argument("--hist", default="8192,16384,32768,65536,131072")
args = ap.parse_args()

ssd_bw, a, b, ref_model = 2.3e9, 0.0012, 4.0e-8, "Llama-3.2-3B"  # M4 defaults (from Exp H)
if args.fit:
    data = json.load(open(args.fit))
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    loads = [(r["kv_gb"] * 1e9, r["load_s"]["single"]) for r in rows]
    ssd_bw = sum(x for x, _ in loads) / sum(t for _, t in loads)
    X, y = [], []
    for r in rows:
        for m, t in r["compute_s"].items():
            X.append([int(m), int(m) * r["hist"]]); y.append(t)
    (a, b), *_ = np.linalg.lstsq(np.array(X, float), np.array(y), rcond=None)
    ref_model = next((k for k in MODELS if k in data["info"]["model"]), ref_model)
    print(f"fitted to {args.fit}: SSD {ssd_bw / 1e9:.2f} GB/s, compute = new*({a * 1e3:.3f} ms + {b * 1e9:.2f} ns*hist)")
    for r in rows:
        for m in r["compute_s"]:
            m_ = int(m)
            load = r["kv_gb"] * 1e9 / ssd_bw
            comp = m_ * (a + b * r["hist"])
            print(f"  check {r['hist']:>6} tok, new {m_:>4}: predicted streamed {max(load, comp):.3f} s "
                  f"vs measured {r['ttft'].get(f'streamed/single/{m}', float('nan')):.3f} s")
if args.ssd:
    ssd_bw = args.ssd * 1e9
budget = args.ram * 0.75
ref_per_tok = MODELS[ref_model][0]
print(f"\nESTIMATE for {args.ram:g} GB RAM (GPU budget ~{budget:.1f} GB), SSD {ssd_bw / 1e9:.1f} GB/s, "
      f"new message {args.new} tokens. Compute scales from {ref_model} by KV size.\n")
print(f"{'model':14} {'hist':>7} {'KV GB':>6} {'fits':>5} {'load s':>7} {'compute s':>9} {'streamed s':>10} {'speedup':>7}")
for name, (per_tok, layers, w) in MODELS.items():
    scale = per_tok / ref_per_tok
    for h in map(int, args.hist.split(",")):
        kv = h * per_tok
        fits = w + 2.2 * kv / 1e9 + 0.5 <= budget
        load = kv / ssd_bw
        comp = args.new * (a + b * h) * scale
        streamed = max(load, comp) + min(load, comp) / layers
        print(f"{name:14} {h:>7} {kv / 1e9:>6.1f} {'yes' if fits else 'no':>5} {load:>7.2f} {comp:>9.2f} "
              f"{streamed:>10.2f} {(load + comp) / streamed:>6.2f}×")
