#!/usr/bin/env python3
"""Generate 3-panel comparison images.

Left : rainy input image
Mid  : model derained output image (from experiments folder)
Right: clean GT image

Pairs are read from a CSV with header columns: clean,rainy

Example:
  python scripts/make_triptych_comparisons.py \
    --pairs_csv gtrain_pairs_val.csv \
    --pred_dir experiments_omgsr_sana_rain \
    --out_dir comparisons_triptych \
    --limit 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


SKIP_TOKEN = "-C-"  # filenames containing this are already clean


@dataclass(frozen=True)
class PairRow:
    clean: str
    rainy: str


class _ProgressBar:
    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = max(0, int(total))
        self.enabled = enabled and sys.stderr.isatty()
        self.current = 0

    def update(self, current: int, *, suffix: str = "") -> None:
        if not self.enabled:
            return
        self.current = max(0, int(current))
        total = max(1, self.total) if self.total > 0 else 0
        if self.total <= 0:
            msg = f"Processed {self.current}{(' ' + suffix) if suffix else ''}"
            sys.stderr.write("\r" + msg.ljust(120))
            sys.stderr.flush()
            return

        width = 30
        frac = min(1.0, self.current / float(total))
        filled = int(round(width * frac))
        bar = "=" * filled + "-" * (width - filled)
        pct = int(round(100 * frac))
        msg = f"[{bar}] {self.current}/{self.total} ({pct:3d}%)"
        if suffix:
            msg += "  " + suffix
        sys.stderr.write("\r" + msg.ljust(120))
        sys.stderr.flush()

    def finish(self, *, suffix: str = "") -> None:
        if self.total > 0:
            self.update(self.total, suffix=suffix)
        else:
            self.update(self.current, suffix=suffix)
        self.close()

    def close(self) -> None:
        if not self.enabled:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


def _read_pairs(csv_path: Path) -> Iterable[PairRow]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        required = {"clean", "rainy"}
        missing = required.difference({name.strip() for name in reader.fieldnames})
        if missing:
            raise ValueError(f"CSV missing columns {sorted(missing)}: {csv_path}")

        for row in reader:
            clean = (row.get("clean") or "").strip()
            rainy = (row.get("rainy") or "").strip()
            if not clean or not rainy:
                continue
            yield PairRow(clean=clean, rainy=rainy)


def _count_pairs(csv_path: Path) -> int:
    return sum(1 for _ in _read_pairs(csv_path))


def _open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
    if height <= 0:
        return img
    w, h = img.size
    if h == height:
        return img
    new_w = max(1, int(round(w * (height / float(h)))))
    return img.resize((new_w, height), resample=Image.BICUBIC)


def _concat_horiz(images: list[Image.Image], gap: int = 0, bg: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    if not images:
        raise ValueError("No images to concat")
    heights = [im.size[1] for im in images]
    if len(set(heights)) != 1:
        raise ValueError("All images must have same height before concat")

    total_w = sum(im.size[0] for im in images) + gap * (len(images) - 1)
    h = images[0].size[1]
    canvas = Image.new("RGB", (total_w, h), color=bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0] + gap
    return canvas


def _safe_stem_from_path(p: str) -> str:
    # Use filename without suffix as output stem.
    # Also guard against weird path separators.
    name = Path(p).name
    return os.path.splitext(name)[0]


def _find_prediction(pred_dir: Path, rainy_path: Path, clean_path: Path) -> Path | None:
    """Find model output image.

    Primary: same basename as rainy image (common inference behavior)
    Fallback: same basename as clean image
    """
    cand1 = pred_dir / rainy_path.name
    if cand1.exists():
        return cand1

    cand2 = pred_dir / clean_path.name
    if cand2.exists():
        return cand2

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate rainy/pred/clean triptych comparisons.")
    parser.add_argument("--pairs_csv", type=Path, required=True, help="CSV file with columns: clean,rainy")
    parser.add_argument("--pred_dir", type=Path, required=True, help="Directory containing model output PNGs")
    parser.add_argument("--out_dir", type=Path, required=True, help="Where to write triptych images")
    parser.add_argument("--limit", type=int, default=0, help="Max number of triptychs to generate (0=all)")
    parser.add_argument("--height", type=int, default=0, help="Resize all panels to this height (0=auto/min)")
    parser.add_argument("--gap", type=int, default=8, help="Gap in pixels between panels")
    parser.add_argument(
        "--skip_token",
        type=str,
        default=SKIP_TOKEN,
        help="If prediction filename contains this token, skip (default: -C-)",
    )
    parser.add_argument(
        "--missing",
        choices=["skip", "error"],
        default="skip",
        help="What to do when any image is missing",
    )
    args = parser.parse_args()

    pairs_csv: Path = args.pairs_csv
    pred_dir: Path = args.pred_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    skipped = 0
    missing = 0
    processed = 0

    total_pairs = _count_pairs(pairs_csv)
    pbar = _ProgressBar(total_pairs, enabled=True)
    stopped_early = False

    for pair in _read_pairs(pairs_csv):
        processed += 1
        rainy_path = Path(pair.rainy)
        clean_path = Path(pair.clean)

        pred_path = _find_prediction(pred_dir, rainy_path, clean_path)
        if pred_path is None:
            missing += 1
            if args.missing == "error":
                raise FileNotFoundError(f"Prediction not found for rainy={rainy_path.name} clean={clean_path.name}")
            pbar.update(processed, suffix=f"wrote={made} skipped={skipped} missing={missing}")
            continue

        # Skip cases where the *prediction file* itself is actually a clean (C) frame.
        if args.skip_token and args.skip_token in pred_path.name:
            skipped += 1
            pbar.update(processed, suffix=f"wrote={made} skipped={skipped} missing={missing}")
            continue

        if not rainy_path.exists() or not clean_path.exists() or not pred_path.exists():
            missing += 1
            if args.missing == "error":
                raise FileNotFoundError(f"Missing file(s): rainy={rainy_path} pred={pred_path} clean={clean_path}")
            pbar.update(processed, suffix=f"wrote={made} skipped={skipped} missing={missing}")
            continue

        rainy_img = _open_rgb(rainy_path)
        pred_img = _open_rgb(pred_path)
        clean_img = _open_rgb(clean_path)

        if args.height > 0:
            target_h = args.height
        else:
            # Avoid upscaling by default: take min height.
            target_h = min(rainy_img.size[1], pred_img.size[1], clean_img.size[1])

        rainy_img = _resize_to_height(rainy_img, target_h)
        pred_img = _resize_to_height(pred_img, target_h)
        clean_img = _resize_to_height(clean_img, target_h)

        triptych = _concat_horiz([rainy_img, pred_img, clean_img], gap=max(0, args.gap), bg=(0, 0, 0))

        out_name = f"{_safe_stem_from_path(pair.rainy)}__triptych.png"
        triptych.save(out_dir / out_name)
        made += 1
        pbar.update(processed, suffix=f"wrote={made} skipped={skipped} missing={missing}")
        if args.limit and made >= args.limit:
            stopped_early = True
            break

    if stopped_early:
        pbar.finish(suffix=f"wrote={made} skipped={skipped} missing={missing}")
    else:
        pbar.finish(suffix=f"wrote={made} skipped={skipped} missing={missing}")

    print(
        f"Done. wrote={made} skipped={skipped} missing={missing} "
        f"out_dir={out_dir} pred_dir={pred_dir} pairs_csv={pairs_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
