"""make.fun cutest-picture-01 entry: writes the picture to /output/entry.png."""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "picture.png")
OUT_DIR = "/output"

os.makedirs(OUT_DIR, exist_ok=True)
shutil.copyfile(SRC, os.path.join(OUT_DIR, "entry.png"))
