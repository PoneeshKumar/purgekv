// Scatter read of one KV-cache tensor segment into per-head destination buffers,
// bypassing the page cache (F_NOCACHE). C++ twin of fastread.read_python, used to
// test whether native code beats Python's os.preadv (which already drops the GIL).
#include <cerrno>
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

extern "C" int pkv_read_scatter(const char* path, long long offset, int n,
                                void** dsts, long long* sizes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -errno;
    fcntl(fd, F_NOCACHE, 1);
    std::vector<iovec> iov(n);
    long long want = 0;
    for (int i = 0; i < n; ++i) {
        iov[i].iov_base = dsts[i];
        iov[i].iov_len = static_cast<size_t>(sizes[i]);
        want += sizes[i];
    }
    long long done = 0;
    int first = 0;
    while (done < want) {
        ssize_t r = preadv(fd, iov.data() + first, n - first, offset + done);
        if (r <= 0) { int e = r < 0 ? errno : EIO; close(fd); return -e; }
        done += r;
        // advance the iovec list past what was read (short reads are rare)
        while (first < n && r >= static_cast<ssize_t>(iov[first].iov_len)) { r -= iov[first].iov_len; ++first; }
        if (first < n && r > 0) {
            iov[first].iov_base = static_cast<char*>(iov[first].iov_base) + r;
            iov[first].iov_len -= r;
        }
    }
    close(fd);
    return 0;
}
