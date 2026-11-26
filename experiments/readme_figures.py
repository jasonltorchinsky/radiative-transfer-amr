"""
Copy experiment figures into an experiment's figures/ directory at a reduced
size, for the README files.

Usage (from the repository root):
    python experiments/readme_figures.py WIDTH SRC_PNG DST_PNG [SRC_PNG DST_PNG ...]

Each source PNG is resized to WIDTH pixels wide (keeping its aspect ratio),
converted to an 8-bit palette and saved with PNG optimisation.
"""

# Standard Library Imports
import os
import sys

# Third-Party Library Imports
from PIL import Image


def shrink(src: str, dst: str, width: int) -> None:
    image: Image.Image = Image.open(src).convert("RGB")
    if image.width > width:
        height: int = round(image.height * width / image.width)
        image = image.resize((width, height), Image.LANCZOS)
    image = image.quantize(colors = 256, method = Image.Quantize.MEDIANCUT)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok = True)
    image.save(dst, optimize = True)


def main() -> None:
    width: int = int(sys.argv[1])
    pairs: list = sys.argv[2:]
    if (len(pairs) == 0) or (len(pairs) % 2 != 0):
        sys.exit(__doc__)
    for src, dst in zip(pairs[0::2], pairs[1::2]):
        shrink(src, dst, width)
        print("{} -> {} ({} bytes)".format(src, dst, os.path.getsize(dst)))


if __name__ == "__main__":
    main()
