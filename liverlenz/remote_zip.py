"""Read members of a remote ZIP over HTTP range requests.

Lets ``scripts/evaluate_chaos.py`` pull just the CT studies it needs from the
890 MB CHAOS archive on Zenodo instead of downloading everything.
"""
from __future__ import annotations

import io
import time
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP (server must support Range)."""

    def __init__(self, url: str, block: int = 1 << 20, retries: int = 5):
        self.url, self.block, self.retries = url, block, retries
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()  # follow redirects once
            self.size = int(r.headers["Content-Length"])
        self._cache: dict[int, bytes] = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        while n > 0:
            bidx = self.pos // self.block
            if bidx not in self._cache:
                if len(self._cache) > 64:
                    self._cache.clear()
                start = bidx * self.block
                self._cache[bidx] = self._fetch(start, min(start + self.block, self.size))
            chunk = self._cache[bidx][self.pos - bidx * self.block:][:n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
