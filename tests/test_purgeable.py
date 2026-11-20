"""Kernel purgeable-memory wrapper (macOS only, no model needed)."""
import os
import numpy as np
import pytest

import purgeable as P
from purgekv.common import phys_footprint_mb

MB = 1 << 20


def test_volatile_round_trip_and_footprint():
    r = P.PurgeableRegion(128 * MB)
    try:
        a = np.frombuffer(r.buffer(), dtype=np.uint8)
        a[:] = 7
        before = phys_footprint_mb(os.getpid())
        r._set(P.VOLATILE | (7 << 8))
        after = phys_footprint_mb(os.getpid())
        assert before - after > 100, "volatile pages should leave phys_footprint"
        if not r.pin():
            pytest.skip("the kernel purged the region (machine under memory pressure)")
        assert int(a[::4096].min()) == 7
    finally:
        r.free()


def test_purged_region_reports_empty():
    r = P.PurgeableRegion(16 * MB)
    try:
        np.frombuffer(r.buffer(), dtype=np.uint8)[:] = 1
        r._set(P.EMPTY)  # ask the kernel to drop it now
        assert r.pin() is False
    finally:
        r.free()
