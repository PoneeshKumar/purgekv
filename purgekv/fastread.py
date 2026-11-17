"""Single-copy layer reader for raw-codec SegmentKVStore files.

Each layer's K (and V) is read with one scatter read per segment straight into a
single pre-sized buffer that already has room for the new tokens. Compared with the
legacy reader this removes the per-segment arrays, the concatenate, and the
full-layer reallocation mlx-lm's KVCache does when the first new token arrives.

Two backends with identical results:
  "single"  Python os.preadv (releases the GIL)
  "native"  the same loop in C++ (purgekv/native/pkvread.cpp), compiled on first use
"""
import ctypes, fcntl, os, subprocess, threading
import numpy as np

F_NOCACHE = 48
STEP = 256  # mlx-lm KVCache grows in 256-token steps; keep capacity on that grid


def capacity(n_ctx, n_new):
    return ((n_ctx + n_new + STEP - 1) // STEP) * STEP


def plan_layer(layout, i):
    """For layer i: list of (path, file_offset, [(head, t0, t1)]) per segment and tensor."""
    k_jobs, v_jobs, t0 = [], [], 0
    for path, ranges, _ in layout:
        (k_off, _, k_shape, _), = ranges[2 * i]
        (v_off, _, v_shape, _), = ranges[2 * i + 1]
        t1 = t0 + k_shape[2]
        k_jobs.append((path, k_off, t0, t1, k_shape))
        v_jobs.append((path, v_off, t0, t1, v_shape))
        t0 = t1
    return k_jobs, v_jobs, t0


def _targets(buf, t0, t1):
    # buf: (1, H, cap, D) uint16; the segment holds (1, H, t1-t0, D) contiguous, i.e.
    # head-major, so each head's slice is one contiguous run in both file and buffer
    return [buf[0, h, t0:t1, :] for h in range(buf.shape[1])]


def read_python(jobs, buf):
    for path, off, t0, t1, _ in jobs:
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
            views = [memoryview(t).cast("B") for t in _targets(buf, t0, t1)]
            want = sum(v.nbytes for v in views)
            got = os.preadv(fd, views, off)
            if got != want:  # rare short read: finish the rest one head at a time
                pos = off
                for v in views:
                    n = 0
                    while n < v.nbytes:
                        r = os.preadv(fd, [v[n:]], pos + n)
                        if r <= 0:
                            raise OSError(f"short read in {path}")
                        n += r
                    pos += v.nbytes
        finally:
            os.close(fd)


# ---------------- native backend ----------------
_lib = None
_lib_lock = threading.Lock()  # I/O threads may all ask for the library at once
_SRC = os.path.join(os.path.dirname(__file__), "native", "pkvread.cpp")


def _native():
    global _lib
    with _lib_lock:
        return _native_locked()


def _native_locked():
    global _lib
    if _lib is None:
        out_dir = os.path.join(os.path.expanduser("~"), ".cache", "purgekv")
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, "libpkvread.dylib")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(_SRC):
            subprocess.run(["clang++", "-O3", "-std=c++17", "-shared", "-fPIC", _SRC, "-o", so], check=True)
        lib = ctypes.CDLL(so)
        lib.pkv_read_scatter.argtypes = [ctypes.c_char_p, ctypes.c_longlong, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_longlong)]
        lib.pkv_read_scatter.restype = ctypes.c_int
        _lib = lib
    return _lib


def read_native(jobs, buf):
    lib = _native()
    for path, off, t0, t1, _ in jobs:
        ts = _targets(buf, t0, t1)
        ptrs = (ctypes.c_void_p * len(ts))(*[t.ctypes.data for t in ts])
        sizes = (ctypes.c_longlong * len(ts))(*[t.nbytes for t in ts])
        rc = lib.pkv_read_scatter(path.encode(), off, len(ts), ptrs, sizes)
        if rc != 0:
            raise OSError(-rc, f"pkv_read_scatter failed for {path}")


def read_layer(layout, i, n_new, backend="single"):
    """Return (k_buf, v_buf, n_ctx) as uint16 numpy arrays of shape (1, H, cap, D)."""
    k_jobs, v_jobs, n_ctx = plan_layer(layout, i)
    _, H, _, Dk = k_jobs[0][4]
    Dv = v_jobs[0][4][3]
    cap = capacity(n_ctx, n_new)
    kb = np.zeros((1, H, cap, Dk), np.uint16)  # zero tail = same as mlx-lm's padding
    vb = np.zeros((1, H, cap, Dv), np.uint16)
    fn = read_native if backend == "native" else read_python
    fn(k_jobs, kb)
    fn(v_jobs, vb)
    return kb, vb, n_ctx
