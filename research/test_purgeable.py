"""Semantics checks for the purgeable layer (no model needed).

1. volatile + group 7 survives with no pressure and data is intact
2. volatile pages leave the process phys_footprint (what jetsam ranks by)
3. group 0 vs group 7: how often a 512 MB object is purged at once on this Mac
"""
import os, time
import numpy as np
from purgekv import purgeable as P
from purgekv.common import phys_footprint_mb

MB = 1 << 20
pid = os.getpid()

r = P.PurgeableRegion(512 * MB)
a = np.frombuffer(r.buffer(), dtype=np.uint8)
a[:] = np.random.default_rng(0).integers(0, 255, a.size, dtype=np.uint8)
chk = int(a[::4096].astype(np.int64).sum())
fp_nonvol = phys_footprint_mb(pid)
r._set(P.VOLATILE | (7 << 8))
time.sleep(0.2)
fp_vol = phys_footprint_mb(pid)
ok = r.pin()
print(f"[1] survived={ok} data_intact={ok and int(a[::4096].astype(np.int64).sum()) == chk}")
print(f"[2] footprint nonvolatile={fp_nonvol:.0f} MB  volatile={fp_vol:.0f} MB  (drop {fp_nonvol - fp_vol:.0f} MB)")
r.free()

for grp in (0, 7):
    purged = 0
    for i in range(10):
        r = P.PurgeableRegion(512 * MB)
        np.frombuffer(r.buffer(), dtype=np.uint8)[::4096] = 1
        np.frombuffer(r.buffer(), dtype=np.uint8)[:] = 1
        r._set(P.VOLATILE | (grp << 8))
        time.sleep(0.1)
        purged += not r.pin()
        r.free()
    print(f"[3] group {grp}: purged {purged}/10 within 100 ms with no deliberate pressure")
