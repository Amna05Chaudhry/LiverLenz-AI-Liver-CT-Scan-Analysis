"""Fetch a few LiTS scans from the Medical Segmentation Decathlon (Task03_Liver, CC BY-SA 4.0).

The MSD liver task contains the LiTS training volumes (liver_N == LiTS volume-N). The 28.9 GB tar
on the MONAI S3 mirror supports HTTP range requests, so we walk the tar headers (512 bytes each)
and download only the requested members into ``data/msd/``.

    python scripts/fetch_msd_examples.py 0 33 37        # LiTS/MSD case ids (test-split patients)
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
OUT = Path(__file__).resolve().parent.parent / "data" / "msd"


def get(start: int, end: int, retries: int = 6) -> bytes:
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))
    raise RuntimeError


def walk(wanted: set[str]):
    """Yield (name, data_offset, size) for tar members until all wanted names are found."""
    off, found, longname = 0, set(), None
    while found != wanted:
        h = get(off, off + 512)
        if h == b"\0" * 512:
            break
        name = longname or h[:100].rstrip(b"\0").decode()
        prefix = h[345:500].rstrip(b"\0").decode()
        if prefix and not longname:
            name = prefix + "/" + name
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        typ = h[156:157]
        data = off + 512
        longname = None
        if typ == b"L":                      # GNU long name: next header uses this name
            longname = get(data, data + size).rstrip(b"\0").decode()
        elif typ in (b"0", b"\0"):
            key = "/".join(name.split("/")[-2:])          # e.g. "labelsTr/liver_0.nii.gz"
            if key in wanted:
                found.add(key)
                yield name, data, size
        off = data + ((size + 511) // 512) * 512


def main(ids: list[int]):
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {f"{d}/liver_{i}.nii.gz" for i in ids for d in ("imagesTr", "labelsTr")}
    print("walking tar headers ...", flush=True)
    for name, off, size in walk(wanted):
        kind = "label" if "/labelsTr/" in name else "image"
        dest = OUT / f"{kind}_{name.split('/')[-1]}"
        if dest.exists() and dest.stat().st_size == size:
            continue
        print(f"downloading {name} ({size / 1e6:.1f} MB)", flush=True)
        buf = bytearray()
        for s in range(off, off + size, 8 << 20):
            buf += get(s, min(s + (8 << 20), off + size))
        dest.write_bytes(bytes(buf))
    print("have:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0])
