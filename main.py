"""make.fun cutest-picture-01 entry: writes the picture to /output/entry.png.

The picture is drawn offline by render.py and committed as picture.png; this
program only checks it and writes its exact bytes, so every run is identical.
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "picture.png")
OUT_DIR = "/output"
MAX_BYTES = 1048576
MAX_SIDE = 1024


def main():
    with open(SRC, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        sys.exit("picture.png is not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    if len(data) > MAX_BYTES or width > MAX_SIDE or height > MAX_SIDE:
        sys.exit("picture.png is over the limits")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "entry.png"), "wb") as f:
        f.write(data)


if __name__ == "__main__":
    main()
