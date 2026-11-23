"""Exp K: purgekv-server with coding agents sharing one server, over real HTTP.

  python research/bench_server.py [--model M] [--agents 4] [--ctx 8192] [--rounds 4]
                                  [--tool-s 1.5] [--ram-gb 1.5] [--configs stock,v03_12,v03_123]

N agents take turns on one server. Each agent has its own long context (a different
real-text "repository" in its first message). A turn: the agent sends its history plus
a ~150-token tool result, streams a short reply (greedy), then runs a "tool" for
--tool-s seconds while the next agent takes its turn. --ram-gb is the server's RAM
budget for cached chats (--prompt-cache-bytes); with 4 agents it holds about one.

Configs (each a fresh server process and a fresh cache directory):
  stock        mlx_lm.server, same RAM budget: a chat pushed out of RAM is recomputed
  v03_1        purgekv-server, SSD tier, no prefetch
  v03_12       purgekv-server, SSD tier + think-time prefetch
  v03_123      purgekv-server, SSD tier + prefetch + 8-bit KV (--kv-bits 8)
  stock_kv8    mlx_lm.server --kv-bits 8 (output reference for v03_123)
  stock_ram    mlx_lm.server with RAM for every chat (output control: mlx-lm's own prompt cache)
After the rounds, each server is restarted and every agent sends one more turn.
Measured client-side: time to first token (TTFT). Round 0 builds the contexts and is
reported separately.
"""
import argparse, json, os, shutil, signal, subprocess, sys, tempfile, time, urllib.request
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
ap.add_argument("--agents", type=int, default=4)
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--tool-s", type=float, default=1.5)
ap.add_argument("--ram-gb", type=float, default=1.5)
ap.add_argument("--prefetch-gb", type=float, default=1.5)
ap.add_argument("--max-tokens", type=int, default=24)
ap.add_argument("--configs", default="stock,v03_1,v03_12,v03_123,stock_kv8")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--out", default=None)
ap.add_argument("--merge", action="store_true", help="keep configs already in --out's file")
args = ap.parse_args()
PY = sys.executable
URL = f"http://127.0.0.1:{args.port}"


def real_text(n_tokens, seed):
    """Real Python source from the standard library, ~n_tokens long, different per seed."""
    import random, sysconfig
    from mlx_lm import tokenizer_utils  # noqa: F401  (ensures mlx_lm is importable)
    lib = sysconfig.get_paths()["stdlib"]
    files = sorted(f for f in os.listdir(lib) if f.endswith(".py"))
    random.Random(seed).shuffle(files)
    out, n = [], 0
    for f in files:
        try:
            s = open(os.path.join(lib, f), encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        out.append(f"# file: {f}\n{s}")
        n += len(s) // 3.4  # ~chars per token for code with this tokenizer
        if n >= n_tokens:
            break
    text = "\n".join(out)
    return text[: int(n_tokens * 3.4)]


def tool_result(agent, rnd):
    tests = ", ".join(f"test_{agent}_{rnd}_{i} PASSED" for i in range(18))
    return f"Tool result (run_tests, round {rnd}): {tests}. 0 failed. What should we do next? Answer in one line."


CONFIGS = {
    "stock": ["-m", "mlx_lm", "server"],
    "stock_kv8": ["-m", "mlx_lm", "server", "--kv-bits", "8", "--quantized-kv-start", "0"],
    "stock_ram": ["-m", "mlx_lm", "server"],
    "v03_1": ["-m", "purgekv.server", "--prefetch-gb", "0"],
    "v03_12": ["-m", "purgekv.server", "--prefetch-gb", str(args.prefetch_gb)],
    "v03_123": ["-m", "purgekv.server", "--prefetch-gb", str(args.prefetch_gb),
                "--kv-bits", "8", "--quantized-kv-start", "0"],
}


def start(cfg, kv_dir, log):
    cmd = [PY, *CONFIGS[cfg], "--model", args.model, "--port", str(args.port),
           "--prompt-cache-bytes", str(int((20 if cfg == "stock_ram" else args.ram_gb) * 1e9))]
    if cfg.startswith("v03"):
        cmd += ["--kv-dir", kv_dir]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            urllib.request.urlopen(URL + "/health", timeout=1)
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError(f"{cfg} server exited; see {log.name}")
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def stop(p):
    p.send_signal(signal.SIGINT)
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()


def turn(messages):
    body = json.dumps({"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
                       "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(c)
    return ttft or (time.perf_counter() - t0), "".join(text), usage


def run(cfg):
    kv_dir = tempfile.mkdtemp(prefix=f"purgekv-{cfg}-", dir=os.environ.get("SPILL_DIR"))
    log = open(f"results/server_{cfg}.log", "w")
    hist = [[{"role": "system", "content": "You are a coding agent. Be brief."},
             {"role": "user", "content": "Here is the repository:\n" + real_text(args.ctx, 100 + a)}]
            for a in range(args.agents)]
    rows, p = [], start(cfg, kv_dir, log)
    try:
        for rnd in range(args.rounds + 1):  # the last round is after a restart
            if rnd == args.rounds:
                stop(p); p = start(cfg, kv_dir, log)
            for a in range(args.agents):
                if rnd > 0:
                    hist[a].append({"role": "user", "content": tool_result(a, rnd)})
                ttft, text, usage = turn(hist[a])
                hist[a].append({"role": "assistant", "content": text})
                row = {"round": rnd, "agent": a, "ttft": round(ttft, 3), "text": text,
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                       "after_restart": rnd == args.rounds}
                rows.append(row)
                print(cfg, json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
                time.sleep(args.tool_s)  # this agent runs its tool; the server is free for others
    finally:
        stop(p)
        log.close()
    stats = {}
    sp = os.path.join(kv_dir, "stats.json")
    if os.path.exists(sp):
        stats = json.load(open(sp))  # counters since the restart
    disk = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(kv_dir) for f in fs)
    shutil.rmtree(kv_dir, ignore_errors=True)
    steady = [r["ttft"] for r in rows if 0 < r["round"] < args.rounds]
    restart = [r["ttft"] for r in rows if r["after_restart"]]
    return {"rows": rows, "stats_after_restart": stats, "disk_gb": round(disk / 1e9, 3),
            "build_ttft_median": round(st.median(r["ttft"] for r in rows if r["round"] == 0), 3),
            "ttft_median": round(st.median(steady), 3), "ttft_p90": round(sorted(steady)[int(0.9 * (len(steady) - 1))], 3),
            "ttft_max": round(max(steady), 3),
            "restart_ttft_median": round(st.median(restart), 3)}


os.makedirs("results", exist_ok=True)
out = {"args": vars(args), "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
       "ram_gb": int(os.popen("sysctl -n hw.memsize").read()) // 2**30, "configs": {}}
path = args.out or f"results/server_{args.model.split('/')[-1]}.json"
if args.merge and os.path.exists(path):  # add configs to an earlier run's file
    out["configs"] = json.load(open(path))["configs"]
for cfg in args.configs.split(","):
    out["configs"][cfg] = run(cfg)
    json.dump(out, open(path, "w"), indent=1)
    c = out["configs"][cfg]
    print(cfg, {k: v for k, v in c.items() if k != "rows"}, flush=True)

# same replies? (greedy decoding; full precision vs stock, 8-bit vs stock 8-bit)
def texts(cfg):
    return [r["text"] for r in out["configs"][cfg]["rows"]] if cfg in out["configs"] else None


for a, b in (("v03_12", "stock_ram"), ("v03_1", "stock"), ("v03_12", "stock"), ("v03_123", "stock_kv8"), ("v03_123", "stock")):
    ta, tb = texts(a), texts(b)
    if ta and tb:
        out.setdefault("identical_replies", {})[f"{a}_vs_{b}"] = f"{sum(x == y for x, y in zip(ta, tb))}/{len(ta)}"
json.dump(out, open(path, "w"), indent=1)
print("identical", out.get("identical_replies"))
print("wrote", path)
