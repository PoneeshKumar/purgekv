"""Stand-in for the user's foreground app: grabs H GB of incompressible memory.

Protocol on stdin/stdout: prints alloc timing, waits for a line, re-touches all
of its memory (timing shows how much of it was pushed to swap meanwhile),
prints that, waits for a line, exits.
"""
import json, sys, time
import numpy as np

gb = float(sys.argv[1])
CH = 256 << 20
block = np.random.default_rng(0).integers(0, 255, CH, dtype=np.uint8)
chunks, t0 = [], time.perf_counter()
for _ in range(int(gb * 1024 / 256)):
    a = np.empty(CH, np.uint8)
    a[:] = block
    a[::4096] = np.random.default_rng(len(chunks)).integers(0, 255, CH // 4096, dtype=np.uint8)
    chunks.append(a)
print(json.dumps({"alloc_s": time.perf_counter() - t0}), flush=True)
for line in sys.stdin:  # each line = re-touch everything once; EOF = exit
    t0, s = time.perf_counter(), 0
    for a in chunks:
        s += int(a[::16384].sum())  # one byte per 16 KB page: pure page-in cost
    print(json.dumps({"retouch_s": time.perf_counter() - t0}), flush=True)
