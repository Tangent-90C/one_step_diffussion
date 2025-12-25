#!/usr/bin/env python3
import argparse
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def list_images(root: Path, out: Path):
    root = root.expanduser().resolve()
    out = out.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")
    paths = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
    paths.sort()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for p in paths:
            f.write(p + '\n')
    print(f"Wrote {len(paths)} paths to {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='Root folder to search')
    parser.add_argument('--out', type=Path, required=True, help='Output txt file')
    args = parser.parse_args()
    list_images(args.root, args.out)
