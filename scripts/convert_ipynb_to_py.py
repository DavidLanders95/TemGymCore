#!/usr/bin/env python3
"""Convert Jupyter notebooks (.ipynb) to plain Python scripts.

This converter intentionally has no dependency on nbformat/jupyter packages, so it can
run in restricted environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _iter_notebooks(paths: Iterable[Path]) -> list[Path]:
    notebooks: list[Path] = []
    for p in paths:
        if p.is_dir():
            notebooks.extend(sorted(p.glob("*.ipynb")))
        elif p.suffix == ".ipynb":
            notebooks.append(p)
    return notebooks


def convert_notebook(nb_path: Path, out_path: Path | None = None) -> Path:
    out_path = out_path or nb_path.with_suffix(".py")
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])

    lines: list[str] = []
    lines.append(f"# Auto-generated from: {nb_path}")
    lines.append("# Conversion keeps code cells and comments out notebook magics/shell lines.")
    lines.append("")

    for idx, cell in enumerate(cells, start=1):
        ctype = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        lines.append(f"# %% [cell {idx} - {ctype}]")

        if ctype == "markdown":
            for md_line in source.splitlines():
                lines.append(f"# {md_line}")
            lines.append("")
            continue

        if ctype == "code":
            for code_line in source.splitlines():
                stripped = code_line.lstrip()
                if stripped.startswith("%") or stripped.startswith("!"):
                    lines.append(f"# {code_line}")
                else:
                    lines.append(code_line)
            lines.append("")
            continue

        lines.append(f"# [unsupported cell type: {ctype}]")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Notebook files and/or directories (directories convert *.ipynb within them)",
    )
    args = parser.parse_args()

    notebooks = _iter_notebooks(args.paths)
    if not notebooks:
        print("No notebooks found to convert.")
        return 1

    for nb in notebooks:
        out = convert_notebook(nb)
        print(f"Converted {nb} -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
