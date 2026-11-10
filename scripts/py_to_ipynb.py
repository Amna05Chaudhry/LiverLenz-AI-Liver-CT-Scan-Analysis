"""Convert a '# %%'-style script into a Jupyter notebook: python scripts/py_to_ipynb.py notebooks/x.py"""
import json, sys
from pathlib import Path

src = Path(sys.argv[1]); cells, cur, kind = [], [], "code"

def flush():
    text = "\n".join(cur).strip("\n")
    if text:
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
        cells.append({"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True),
                      **({"outputs": [], "execution_count": None} if kind == "code" else {})})

for line in src.read_text(encoding="utf-8").splitlines():
    if line.startswith("# %%"):
        flush(); cur = []; kind = "markdown" if "[markdown]" in line else "code"
    else:
        cur.append(line)
flush()
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = src.with_suffix(".ipynb"); out.write_text(json.dumps(nb, indent=1), encoding="utf-8"); print(f"wrote {out} ({len(cells)} cells)")
