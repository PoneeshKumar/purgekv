"""Thin ctypes wrapper over XNU purgeable virtual memory.

A purgeable region marked VOLATILE may be discarded by the kernel at any
moment under memory pressure, with no swap write and no process kill. When we
mark it NONVOLATILE again the kernel tells us whether the contents survived.
Volatile pages are also excluded from the process phys_footprint, which is
what jetsam uses to choose a victim.
"""
import ctypes
import ctypes.util

_libc = ctypes.CDLL(ctypes.util.find_library("System"), use_errno=True)

VM_FLAGS_ANYWHERE = 0x1
VM_FLAGS_PURGABLE = 0x2
VM_PURGABLE_SET_STATE = 0
VM_PURGABLE_GET_STATE = 1
NONVOLATILE, VOLATILE, EMPTY = 0, 1, 2

_task = ctypes.c_uint32.in_dll(_libc, "mach_task_self_").value

_libc.mach_vm_allocate.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_int]
_libc.mach_vm_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]
_libc.mach_vm_purgable_control.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
for f in (_libc.mach_vm_allocate, _libc.mach_vm_deallocate, _libc.mach_vm_purgable_control):
    f.restype = ctypes.c_int


def _check(kr, what):
    if kr != 0:
        raise OSError(f"{what} failed: kern_return={kr}")


class PurgeableRegion:
    """One kernel purgeable VM object. The kernel purges it all-or-nothing."""

    def __init__(self, nbytes: int):
        self.size = max(int(nbytes), 1)
        addr = ctypes.c_uint64(0)
        _check(_libc.mach_vm_allocate(_task, ctypes.byref(addr), self.size,
                                      VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE), "mach_vm_allocate")
        self.addr = addr.value
        self.alive = True

    def _set(self, state: int) -> int:
        s = ctypes.c_int(state)
        _check(_libc.mach_vm_purgable_control(_task, self.addr, VM_PURGABLE_SET_STATE, ctypes.byref(s)),
               "mach_vm_purgable_control")
        return s.value  # previous state

    def state(self) -> int:
        s = ctypes.c_int(0)
        _check(_libc.mach_vm_purgable_control(_task, self.addr, VM_PURGABLE_GET_STATE, ctypes.byref(s)),
               "mach_vm_purgable_control")
        return s.value & 0x3

    def buffer(self):
        """Writable memoryview over the region (only valid while NONVOLATILE)."""
        return (ctypes.c_char * self.size).from_address(self.addr)

    def make_volatile(self):
        self._set(VOLATILE)

    def pin(self) -> bool:
        """Mark NONVOLATILE. Returns False if the kernel already purged it."""
        return (self._set(NONVOLATILE) & 0x3) != EMPTY

    def free(self):
        if self.alive:
            _libc.mach_vm_deallocate(_task, self.addr, self.size)
            self.alive = False

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass
