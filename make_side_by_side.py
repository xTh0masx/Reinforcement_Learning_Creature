"""Combine two rendered PNG sequences into one side-by-side MP4 (left | right).

    python make_side_by_side.py --left renders/showcase/myrranox --right renders/showcase/dc \
        --out docs/videos/showcase_side_by_side.mp4
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", default="renders/showcase/myrranox")
    p.add_argument("--right", default="renders/showcase/dc")
    p.add_argument("--out", default="docs/videos/showcase_side_by_side.mp4")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    left = sorted(glob.glob(os.path.join(args.left, "frame_*.png")))
    right = sorted(glob.glob(os.path.join(args.right, "frame_*.png")))
    names = sorted(set(map(os.path.basename, left)) & set(map(os.path.basename, right)))
    names = [n for n in names if os.path.getsize(os.path.join(args.left, n)) and os.path.getsize(os.path.join(args.right, n))]
    print(f"{len(names)} frame pairs")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    w = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    for i, n in enumerate(names):
        a = imageio.imread(os.path.join(args.left, n))[..., :3]
        b = imageio.imread(os.path.join(args.right, n))[..., :3]
        gap = np.full((a.shape[0], 4, 3), 40, dtype=np.uint8)
        w.append_data(np.concatenate([a, gap, b], axis=1))
        if i % 1000 == 0:
            print(f"  {i}/{len(names)}", flush=True)
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
