import ctypes, re, subprocess, time
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

CHUNK = 2048


def fake_prompt(n, seed):
    return np.random.default_rng(seed).integers(1000, 30000, n).tolist()


_CORPUS = None


def real_text_tokens(tokenizer, n, session):
    """n tokens of real English/code text, a different slice per session
    (Python stdlib sources + docstrings, which ship on every Mac with Python)."""
    global _CORPUS
    if _CORPUS is None:
        import glob, os, sysconfig
        files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
        _CORPUS = "\n".join(open(f, errors="ignore").read() for f in files)
    chars = n * 5
    start = (session * 7919 * chars) % (len(_CORPUS) - chars * 2)
    ids = tokenizer.encode(_CORPUS[start:start + chars * 2])
    assert len(ids) >= n
    return ids[:n]


def prefill(model, tokens, cache=None):
    cache = cache if cache is not None else make_prompt_cache(model)
    toks = mx.array(tokens)[None]
    logits = None
    for i in range(0, toks.shape[1], CHUNK):
        logits = model(toks[:, i:i + CHUNK], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return cache, logits[:, -1, :]


def greedy(model, cache, logits, n):
    out = []
    for _ in range(n):
        t = mx.argmax(logits, axis=-1)
        out.append(t.item())
        logits = model(t[None], cache=cache)[:, -1, :]
    return out


def vm_counters():
    s = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", s).group(1))
    page = 16384
    return {"swapouts_mb": get("Swapouts") * page / 2**20,
            "swapins_mb": get("Swapins") * page / 2**20,
            "compressions_mb": get("Compressions") * page / 2**20,
            "purged_mb": get("Pages purged") * page / 2**20}


def swap_used_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", s).group(1))


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
_libc = ctypes.CDLL("/usr/lib/libSystem.dylib")


def pressure_level():
    """macOS memory pressure: 1 normal, 2 warning, 4 critical."""
    v, n = ctypes.c_int(0), ctypes.c_size_t(4)
    _libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(v), ctypes.byref(n), None, 0)
    return v.value


def phys_footprint_mb(pid):
    """ri_phys_footprint from proc_pid_rusage: the number jetsam ranks processes by."""
    buf = (ctypes.c_uint64 * 32)()
    if _libproc.proc_pid_rusage(int(pid), 0, ctypes.byref(buf)) != 0:
        return float("nan")
    return buf[2 + 7] / 2**20  # 16-byte uuid = 2 u64s, then field #7


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
