#!/usr/bin/env python3
"""Construct GT-Rain (GT-RAIN_train) clean/rainy CSV pairs.

Expected filename pattern (common in GT-Rain):
  <prefix>-C-<frame>.<ext>  (clean)
  <prefix>-R-<frame>.<ext>  (rainy)

Example:
  Albergo_0-0-Webcam-C-000.png
  Albergo_0-0-Webcam-R-007.png

Output CSV has 2 columns:
  clean,rainy

Pairing rules per <prefix> group:
- If a clean image exists with the same frame id as a rainy image, pair those.
- Else, if exactly one clean exists in the group, pair that single clean with all rainy frames.
- Else (multiple cleans, no exact match), pick the smallest-frame clean as fallback.

This makes the script robust to both "one clean reference" and "per-frame clean" layouts.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<tag>[CR])-(?P<frame>\d+)\.(?P<ext>[^.]+)$")

# Common image extensions. We default to these to avoid accidentally scanning arbitrary files.
DEFAULT_IMAGE_EXTS: List[str] = [
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "webp",
    "tif",
    "tiff",
]


@dataclass(frozen=True)
class Item:
    path: Path
    prefix: str
    tag: str  # 'C' or 'R'
    frame: int


def _iter_items(root: Path, exts: Optional[Iterable[str]]) -> Iterable[Item]:
    root = root.resolve()
    if exts is None:
        allowed_exts = {e.lower() for e in DEFAULT_IMAGE_EXTS}
    else:
        allowed_exts = {e.lower().lstrip(".") for e in exts}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = _PATTERN.match(p.name)
        if not m:
            continue
        ext = m.group("ext").lower()
        if allowed_exts is not None and ext not in allowed_exts:
            continue
        yield Item(
            path=p,
            prefix=m.group("prefix"),
            tag=m.group("tag"),
            frame=int(m.group("frame")),
        )


def _choose_clean_for_rainy(
    clean_by_frame: Dict[int, Path],
    clean_fallback: Optional[Path],
    rainy_frame: int,
) -> Optional[Path]:
    if rainy_frame in clean_by_frame:
        return clean_by_frame[rainy_frame]
    return clean_fallback


def build_pairs(root: Path, exts: Optional[List[str]] = None) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    """Return (pairs, warnings)."""
    items = list(_iter_items(root, exts))

    groups: Dict[str, Dict[str, List[Item]]] = {}
    for it in items:
        groups.setdefault(it.prefix, {}).setdefault(it.tag, []).append(it)

    pairs: List[Tuple[Path, Path]] = []
    warnings: List[str] = []

    for prefix, by_tag in sorted(groups.items(), key=lambda kv: kv[0]):
        cleans = sorted(by_tag.get("C", []), key=lambda x: x.frame)
        rainies = sorted(by_tag.get("R", []), key=lambda x: x.frame)

        if not cleans:
            if rainies:
                warnings.append(f"[WARN] prefix '{prefix}': has rainy but no clean; skipped {len(rainies)} rainy files")
            continue
        if not rainies:
            continue

        clean_by_frame = {c.frame: c.path for c in cleans}
        clean_fallback: Optional[Path]
        if len(cleans) == 1:
            clean_fallback = cleans[0].path
        else:
            # Multiple clean frames exist: fallback to the smallest frame.
            clean_fallback = cleans[0].path

        # Build pairs
        missing_exact = 0
        for r in rainies:
            cpath = _choose_clean_for_rainy(clean_by_frame, clean_fallback, r.frame)
            if cpath is None:
                continue
            if r.frame not in clean_by_frame:
                missing_exact += 1
            pairs.append((cpath, r.path))

        if len(cleans) > 1 and missing_exact > 0:
            warnings.append(
                f"[WARN] prefix '{prefix}': {len(cleans)} clean files, {len(rainies)} rainy files; "
                f"{missing_exact} rainy frames had no exact clean match (used fallback '{clean_fallback.name}')"
            )

    # Deterministic ordering: sort by rainy then clean
    pairs.sort(key=lambda t: (str(t[1]), str(t[0])))
    return pairs, warnings


def _to_out_path(p: Path, root: Path, relative: bool) -> str:
    if relative:
        return p.resolve().relative_to(root.resolve()).as_posix()
    return p.resolve().as_posix()


def write_csv(
    pairs: List[Tuple[Path, Path]],
    out_csv: Path,
    root: Path,
    relative: bool,
    header: bool,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["clean", "rainy"])
        for c, r in pairs:
            w.writerow([_to_out_path(c, root, relative), _to_out_path(r, root, relative)])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build GT-Rain clean/rainy CSV pairs (2 columns: clean,rainy).",
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root directory (e.g. .../GT-RAIN_train)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path, or a directory (will write gtrain_pairs.csv inside)",
    )
    ap.add_argument(
        "--ext",
        action="append",
        default=None,
        choices=DEFAULT_IMAGE_EXTS,
        help=(
            "Allowed image extension (repeatable), e.g. --ext png --ext jpg. "
            "Default: common image extensions."
        ),
    )
    ap.add_argument(
        "--relative",
        action="store_true",
        help="Write paths relative to --root (default: absolute paths)",
    )
    ap.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write CSV header row",
    )
    ap.add_argument(
        "--print-warnings",
        action="store_true",
        help="Print pairing warnings to stderr",
    )

    args = ap.parse_args()

    root: Path = args.root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--root not found or not a directory: {root}")

    out_csv: Path = args.output
    # If user passes a directory (e.g. --output ./), write a default filename inside.
    if out_csv.exists() and out_csv.is_dir():
        out_csv = out_csv / "gtrain_pairs.csv"
    # If output ends with a separator but doesn't exist, treat it as a directory intent.
    # (argparse Path loses the trailing '/', so we only handle the common existing-dir case.)
    if out_csv.parent.exists() and out_csv.parent.is_dir() and out_csv.name in {"", "."}:
        out_csv = out_csv / "gtrain_pairs.csv"

    pairs, warnings = build_pairs(root=root, exts=args.ext)
    write_csv(
        pairs=pairs,
        out_csv=out_csv,
        root=root,
        relative=bool(args.relative),
        header=not bool(args.no_header),
    )

    if args.print_warnings and warnings:
        import sys

        for w in warnings:
            print(w, file=sys.stderr)

    print(f"Wrote {len(pairs)} pairs to: {out_csv}")
    if warnings and not args.print_warnings:
        print(f"Warnings: {len(warnings)} (use --print-warnings to view)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
