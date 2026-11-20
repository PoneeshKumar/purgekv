"""Five agents, one local model: design, build, check, backtest and paper-trade a strategy.

  python examples/trading_agents.py                      # purgekv-server vs stock mlx_lm.server
  python examples/trading_agents.py --servers purgekv    # just one
  python examples/trading_agents.py --model mlx-community/Qwen3-8B-4bit --rounds 3 --ram-gb 2

A real multi-agent workload for purgekv: five agents (researcher, engineer, QA, backtester,
executor) take turns on ONE local model behind an OpenAI-compatible server. Each keeps its
own long conversation, so on a 16 GB Mac they cannot all stay in the server's RAM budget
(--ram-gb). Every turn prints its time to first token (TTFT) and, on purgekv-server, whether
the agent's chat came from RAM, from SSD, or from a prefetch done while other agents worked.

Safety:
  * The executor only PAPER-trades: it writes the orders it would place to paper_orders.json.
    Nothing here talks to a broker. Backtests are not investment advice.
  * Model-written strategy code runs in a separate Python process with a time limit, inside
    macOS sandbox-exec with network access denied and file writes limited to a temp folder.
  * Prices are synthetic by default (regime-switching random walks, made offline). Pass
    --prices FILE.csv (date,ticker,close,volume) to use your own data.
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, tempfile, textwrap, time, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
ap.add_argument("--servers", default="stock,purgekv", help="comma list: stock, purgekv (run in this order)")
ap.add_argument("--rounds", type=int, default=3, help="research -> build -> QA -> backtest -> execute cycles")
ap.add_argument("--ram-gb", type=float, default=2.0, help="server RAM budget for cached chats")
ap.add_argument("--prefetch-gb", type=float, default=2.0)
ap.add_argument("--kv-bits", type=int, default=None, help="8: store/compute the KV cache in 8-bit")
ap.add_argument("--prices", default=None, help="CSV with date,ticker,close,volume (default: synthetic)")
ap.add_argument("--port", type=int, default=8790)
ap.add_argument("--out", default="results/trading_agents")
args = ap.parse_args()
URL = f"http://127.0.0.1:{args.port}"
os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- data
def synthetic_prices(n_days=2520, tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"), seed=7):
    """Ten years of daily prices: a shared market factor, two regimes (trending / choppy),
    per-stock momentum and short-term mean reversion, volume that rises with volatility."""
    rng = np.random.default_rng(seed)
    T, N = n_days, len(tickers)
    regime = np.zeros(T, int)
    for t in range(1, T):
        regime[t] = regime[t - 1] if rng.random() > 0.01 else 1 - regime[t - 1]
    mkt = rng.normal(0.0003, 0.009, T) * np.where(regime == 1, 1.6, 1.0)
    beta = rng.uniform(0.6, 1.4, N)
    rets = np.zeros((T, N))
    for t in range(1, T):
        mom = 0.004 * rets[max(0, t - 20):t].sum(0) if regime[t] == 0 else 0
        rev = -0.03 * rets[t - 1] if regime[t] == 1 else 0
        rets[t] = beta * mkt[t] + mom + rev + rng.normal(0, 0.016, N)
    close = 50 * np.exp(np.cumsum(rets, 0))
    vol = (1e6 * np.exp(rng.normal(0, 0.3, (T, N)))) * (1 + 20 * np.abs(rets))
    return close.round(2), vol.round(0), list(tickers)


def load_prices():
    if not args.prices:
        return synthetic_prices()
    rows = [line.strip().split(",") for line in open(args.prices)][1:]
    tick = sorted({r[1] for r in rows}); dates = sorted({r[0] for r in rows})
    di, ti = {d: i for i, d in enumerate(dates)}, {t: i for i, t in enumerate(tick)}
    close = np.full((len(dates), len(tick)), np.nan); vol = np.zeros_like(close)
    for d, t, c, v in rows:
        close[di[d], ti[t]] = float(c); vol[di[d], ti[t]] = float(v)
    keep = ~np.isnan(close).any(1)
    return close[keep], vol[keep], tick


CLOSE, VOLUME, TICKERS = load_prices()
DATA = os.path.join(args.out, "prices.npz")
np.savez(DATA, close=CLOSE, volume=VOLUME, tickers=np.array(TICKERS))

# ---------------------------------------------------------------- sandboxed backtest tool
HARNESS = r'''
import json, sys, numpy as np
d = np.load(sys.argv[1]); close, volume = d["close"], d["volume"]
ns = {"np": np}
exec(open(sys.argv[2]).read(), ns)
strategy = ns["strategy"]

def positions(c, v):
    p = np.asarray(strategy(c.copy(), v.copy()), float)
    if p.shape != c.shape:
        raise ValueError(f"strategy must return shape {c.shape}, got {p.shape}")
    return np.clip(np.nan_to_num(p), -1, 1)

pos = positions(close, volume)
cut = int(len(close) * 0.7)
lookahead = not np.allclose(positions(close[:cut], volume[:cut]), pos[:cut], atol=1e-9)
rets = np.diff(close, axis=0) / close[:-1]
held = pos[:-1] / np.maximum(1, np.abs(pos[:-1]).sum(1, keepdims=True))   # gross exposure <= 1
turn = np.abs(np.diff(held, axis=0, prepend=0)).sum(1)
daily = (held * rets).sum(1) - 0.0005 * turn                               # 5 bps per unit traded
def stats(r):
    eq = np.cumprod(1 + r)
    return {"sharpe": round(float(r.mean() / (r.std() + 1e-12) * 252 ** 0.5), 2),
            "cagr": round(float(eq[-1] ** (252 / len(r)) - 1), 4),
            "max_drawdown": round(float((eq / np.maximum.accumulate(eq) - 1).min()), 4)}
print(json.dumps({"train": stats(daily[:cut]), "test": stats(daily[cut:]),
                  "avg_daily_turnover": round(float(turn.mean()), 3),
                  "lookahead_detected": bool(lookahead),
                  "last_positions": [round(float(x), 3) for x in pos[-1]]}))
'''

SANDBOX = '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath "{tmp}"))'


def run_backtest(code):
    tmp = tempfile.mkdtemp(prefix="pkv-bt-")
    try:
        open(os.path.join(tmp, "strategy.py"), "w").write(code)
        open(os.path.join(tmp, "harness.py"), "w").write(HARNESS)
        shutil.copy(DATA, os.path.join(tmp, "prices.npz"))
        prof = SANDBOX.format(tmp=os.path.realpath(tmp))
        cmd = ["sandbox-exec", "-p", prof, sys.executable, "-I", "harness.py", "prices.npz", "strategy.py"]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return {"error": (r.stderr or r.stdout)[-1500:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"error": "timed out after 60 s"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- agents
TASK = ("I want you to make and backtest a trading strategy. There should be one agent that thinks of "
        "different ideas using different factors and available data, then a different agent that creates "
        "the pipeline and algorithms, then a QA agent and a backtester agent that validate the strategy, "
        "and finally an agent that executes the trade.")
BRIEF = f"""
Shared project brief (every agent has this):
- Data: daily close prices and volumes for {len(TICKERS)} stocks ({', '.join(TICKERS)}), {len(CLOSE)} days,
  as numpy arrays `close` and `volume` of shape (days, stocks). Day 0 is the oldest row.
- The strategy is ONE Python function using only numpy:
      def strategy(close, volume): -> positions, same shape, values in [-1, 1]
  Row t may only use data from rows 0..t (no look-ahead). Positions in row t are held from the
  close of day t to the close of day t+1. The backtest normalises gross exposure to <= 1 and
  charges 5 bps per unit traded. First 70% of days = train, last 30% = test (out of sample).
- Backtest harness source (the backtester runs exactly this):
{textwrap.indent(HARNESS.strip(), '    ')}
- Execution is PAPER ONLY: the executor writes orders for a $100,000 paper account to a file.
  There is no broker connection.
"""
ROLES = {
    "researcher": "You are the RESEARCH agent. Propose one concrete, testable strategy idea per turn using "
                  "factors you can build from the data (momentum, mean reversion, volatility, volume, "
                  "cross-sectional ranks, regimes...). Give the exact rules. Learn from earlier backtests "
                  "and try a different factor family when one fails. Max 150 words.",
    "engineer": "You are the ENGINEERING agent. Turn the researcher's idea into the strategy function. Reply "
                "with ONLY one ```python code block defining strategy(close, volume). numpy only, vectorised, "
                "no look-ahead, handle the first rows (warm-up) with zeros.",
    "qa": "You are the QA agent. Review the engineer's code for look-ahead bias, shape errors, NaNs, division "
          "by zero and whether it matches the idea. Reply 'APPROVE' or 'FIX:' followed by the exact fixes. "
          "Max 120 words.",
    "backtester": "You are the BACKTEST agent. You receive the harness output. Judge it: is the out-of-sample "
                  "(test) result good enough, is it overfit (train >> test), is turnover sane, was look-ahead "
                  "detected? Reply 'PASS' or 'FAIL' and three short reasons for the researcher.",
    "executor": "You are the EXECUTION agent. PAPER TRADING ONLY. Given the approved strategy's last positions "
                "and prices, list the paper orders (ticker, side, shares) for a $100,000 paper account and one "
                "line of risk notes. Never claim an order was sent to a real broker.",
}
SYSTEM = {r: f"{TASK}\n\n{BRIEF}\n\n{p}" for r, p in ROLES.items()}


def chat(messages, max_tokens):
    body = json.dumps({"model": args.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0, ttft, out, usage = time.perf_counter(), None, [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    ttft = ttft or time.perf_counter() - t0
                    out.append(c)
    return "".join(out), ttft or time.perf_counter() - t0, time.perf_counter() - t0, usage


class Agent:
    def __init__(self, role, run):
        self.role, self.run, self.msgs = role, run, [{"role": "system", "content": SYSTEM[role]}]

    def ask(self, content, max_tokens=400):
        self.msgs.append({"role": "user", "content": content})
        before = self.run.stats()
        text, ttft, total, usage = chat(self.msgs, max_tokens)
        self.msgs.append({"role": "assistant", "content": text})
        source = self.run.source(before, usage)
        row = {"agent": self.role, "ttft_s": round(ttft, 2), "total_s": round(total, 1),
               "prompt_tokens": usage.get("prompt_tokens"),
               "reused_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "kv_from": source}
        self.run.log.append(row)
        print(f"  {self.role:<11} TTFT {ttft:6.2f}s  context {row['prompt_tokens'] or 0:>6} tok  "
              f"reused {row['reused_tokens'] or 0:>6}  KV from: {source}", flush=True)
        return text


def size_orders(last, px, account=100_000):
    """The orders to actually record, computed in code like the backtest: gross exposure
    = account, split by |position|. A small model's own arithmetic is not trusted."""
    gross = sum(abs(p) for p in last.values()) or 1
    out = []
    for t, p in last.items():
        shares = int(abs(p) / gross * account // px[t])
        if shares:
            out.append({"ticker": t, "side": "BUY" if p > 0 else "SELL SHORT", "shares": shares,
                        "notional": round(shares * px[t], 2)})
    return out


def agent_overspent(text, px, account=100_000):
    """Rough check of the executor's own orders: total notional it implies."""
    total = 0.0
    for t, price in px.items():
        m = re.search(rf"{t}\D{{0,40}}?(\d[\d,]*)\s*shares", text)
        if m:
            total += int(m.group(1).replace(",", "")) * price
    return round(total / account, 2)


def says(text, word):
    """Does the reply start with this verdict? Models often write it in bold: **PASS**."""
    return text.strip().lstrip("*#_ ").upper().startswith(word)


def code_of(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else text


# ---------------------------------------------------------------- servers
class Run:
    def __init__(self, kind):
        self.kind, self.log = kind, []
        self.kv_dir = tempfile.mkdtemp(prefix="pkv-trading-")
        cmd = ([sys.executable, "-m", "purgekv.server", "--kv-dir", self.kv_dir, "--prefetch-gb", str(args.prefetch_gb)]
               if kind == "purgekv" else [sys.executable, "-m", "mlx_lm", "server"])
        cmd += ["--model", args.model, "--port", str(args.port), "--prompt-cache-bytes", str(int(args.ram_gb * 1e9))]
        if args.kv_bits:
            cmd += ["--kv-bits", str(args.kv_bits), "--quantized-kv-start", "0"]
        self.logf = open(os.path.join(args.out, f"server_{kind}.log"), "w")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT)
        for _ in range(1200):
            try:
                urllib.request.urlopen(URL + "/health", timeout=1); return
            except Exception:
                if self.proc.poll() is not None:
                    raise SystemExit(f"{kind} server exited; see {self.logf.name}")
                time.sleep(0.5)

    def stats(self):
        try:
            return json.load(open(os.path.join(self.kv_dir, "stats.json")))
        except (OSError, ValueError):
            return {}

    def source(self, before, usage):
        reused = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        if self.kind == "purgekv":
            time.sleep(0.05)
            after = self.stats()
            d = lambda k: after.get(k, 0) - before.get(k, 0)
            if d("prefetch_hits"):
                return "SSD, prefetched"
            if d("prefetch_partial"):
                return "SSD, part-prefetched"
            if d("ssd_loads"):
                return "SSD"
        prompt = usage.get("prompt_tokens") or 1
        return "RAM" if reused >= 0.5 * prompt else "recomputed"  # reusing only the shared brief = recompute

    def close(self):
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.logf.close()
        shutil.rmtree(self.kv_dir, ignore_errors=True)


def workflow(run):
    A = {r: Agent(r, run) for r in ROLES}
    feedback, results = "Start.", []
    for rnd in range(1, args.rounds + 1):
        print(f"\n[{run.kind}] round {rnd}", flush=True)
        idea = A["researcher"].ask(f"Backtest feedback so far: {feedback}\nPropose the next strategy idea.", 250)
        code = code_of(A["engineer"].ask(f"Implement this idea:\n{idea}", 700))
        review = A["qa"].ask(f"Idea:\n{idea}\n\nCode:\n```python\n{code}\n```", 250)
        if not says(review, "APPROVE"):
            code = code_of(A["engineer"].ask(f"QA review:\n{review}\nReturn the corrected code.", 700))
        bt = run_backtest(code)
        if "error" in bt:  # the harness crashed: one fix attempt, like a real team would
            code = code_of(A["engineer"].ask(f"The backtest crashed:\n{bt['error'][-800:]}\nReturn the fixed code.", 700))
            bt = run_backtest(code)
        verdict = A["backtester"].ask(f"Strategy code:\n```python\n{code}\n```\nHarness output:\n{json.dumps(bt)}", 250)
        orders = None
        if says(verdict, "PASS") and "last_positions" in bt:
            last = dict(zip(TICKERS, bt["last_positions"]))
            px = dict(zip(TICKERS, CLOSE[-1].round(2).tolist()))
            orders = A["executor"].ask(f"Last positions: {json.dumps(last)}\nLast prices: {json.dumps(px)}", 300)
            sized, used = size_orders(last, px), agent_overspent(orders, px)
            if used > 1.05:
                print(f"  risk check: the executor's orders add up to {used}x the paper account; "
                      "recording the code-sized orders instead", flush=True)
            with open(os.path.join(args.out, "paper_orders.json"), "a") as f:
                f.write(json.dumps({"server": run.kind, "round": rnd, "orders": sized, "positions": last,
                                    "prices": px, "executor_text": orders, "executor_notional_x_account": used,
                                    "paper_only": True}) + "\n")
        results.append({"round": rnd, "idea": idea, "code": code, "qa": review, "backtest": bt,
                        "verdict": verdict, "paper_orders": orders})
        feedback = f"round {rnd}: {json.dumps(bt)[:600]} verdict: {verdict[:300]}"
    return results


report = {"model": args.model, "ram_gb": args.ram_gb, "kv_bits": args.kv_bits, "runs": {}}
for kind in args.servers.split(","):
    print(f"\n=== {kind} server: {args.model}, RAM budget {args.ram_gb} GB ===", flush=True)
    run = Run(kind)
    try:
        t0 = time.perf_counter()
        res = workflow(run)
        report["runs"][kind] = {"turns": run.log, "results": res, "wall_s": round(time.perf_counter() - t0, 1)}
    finally:
        run.close()
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)

print("\n=== time to first token per agent turn ===")
kinds = list(report["runs"])
print("turn  agent        " + "".join(f"{k:>26}" for k in kinds))
n = max(len(report["runs"][k]["turns"]) for k in kinds)
for i in range(n):
    cells = []
    for k in kinds:
        t = report["runs"][k]["turns"]
        cells.append(f"{t[i]['ttft_s']:>7.2f}s  {t[i]['kv_from']:<16}" if i < len(t) else " " * 26)
    agent = next(report["runs"][k]["turns"][i]["agent"] for k in kinds if i < len(report["runs"][k]["turns"]))
    print(f"{i + 1:>4}  {agent:<11}" + "".join(f"{c:>26}" for c in cells))
for k in kinds:
    t = [x["ttft_s"] for x in report["runs"][k]["turns"]]
    print(f"{k}: total waiting for first tokens {sum(t):.1f}s over {len(t)} turns, "
          f"whole workflow {report['runs'][k]['wall_s']}s")
print(f"\nreport: {args.out}/report.json   paper orders: {args.out}/paper_orders.json (paper only)")
