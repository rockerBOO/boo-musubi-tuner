"""Labelled contact sheets for the mottle evals (needs opencv-python-headless, numpy, pillow).

Usage: python scripts/mottle_contact_sheet.py --dir IMAGES OUT.png "TITLE" grad|scene  label=tag[|note] ...
  Images are read as IMAGES/{prefix}_{tag}_s{seed}_00001_.png, prefix T1 (grad) or T2 (scene); override with --grad-prefix / --scene-prefix.
  label = row name shown on the sheet, tag = file tag (e.g. turbo, plain200); note = optional 2nd line.
grad : rows = runs, columns = seeds 722-727. Top block = full image; bottom block = high-pass view (x - blur12, x6, 256 px crop):
       grey = clean, coloured blotches = mottle.
scene: rows = runs, seed 722 only. Columns = sky crop, sky crop with contrast x6 (blotches), denim crop (detail check).
"""

import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

O = ""  # image directory, set from --dir
PREFIX = {"grad": "T1", "scene": "T2"}
F = ImageFont.load_default(size=14)
FS = ImageFont.load_default(size=11)
LW, HH = 190, 22  # label column width, header height


def txt(w, h, lines, size=None):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    y = 3
    for i, t in enumerate(lines):
        font = F if i == 0 else FS
        cur = ""
        for word in t.split():  # wrap to the label column
            if d.textlength((cur + " " + word).strip(), font=font) > w - 8:
                d.text((4, y), cur, fill="black" if i == 0 else "#555", font=font)
                y += 18 if i == 0 else 14
                cur = word
            else:
                cur = (cur + " " + word).strip()
        d.text((4, y), cur, fill="black" if i == 0 else "#555", font=font)
        y += 18 if i == 0 else 14
    return np.asarray(im)


def hdr(w, t):
    im = Image.new("RGB", (w, HH), "#ddd")
    ImageDraw.Draw(im).text((4, 3), t, fill="black", font=FS)
    return np.asarray(im)


def load(prefix, l, s):
    return np.asarray(Image.open(f"{O}{prefix}_{l}_s{s}_00001_.png").convert("RGB")).astype(np.float32)


def hp(a):
    return np.clip(np.stack([a[..., c] - cv2.GaussianBlur(a[..., c], (0, 0), 12) for c in range(3)], -1) * 6 + 128, 0, 255)


def grad(rows, seeds=range(722, 728)):
    w, h = 150, 246
    blocks = []
    for name, title in (
        ("full", "Full image (gradient prompt, 800x1312)"),
        ("hp", "High-pass x6, 256 px crop: grey = clean, blotches = mottle"),
    ):
        top = np.concatenate(
            [txt(LW, HH, [title])[:HH] if False else np.full((HH, LW, 3), 221, np.uint8)] + [hdr(w, f"seed {s}") for s in seeds], 1
        )
        body = []
        for lab, tag, note in rows:
            cells = []
            for s in seeds:
                a = load(PREFIX["grad"], tag, s)
                cells.append(
                    cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
                    if name == "full"
                    else cv2.resize(hp(a)[500:756, 200:456], (w, h))
                )
            body.append(np.concatenate([txt(LW, h, [lab, note]), *cells], 1))
        blocks.append((title, np.concatenate([top, *body], 0)))
    return blocks


def scene(rows, seed=722):
    cw, chh = 330, 132
    heads = ["sky crop 1:1", "sky crop, contrast x6 (blotches)", "denim crop 1:1 (detail check)"]
    top = np.concatenate([np.full((HH, LW, 3), 221, np.uint8)] + [hdr(cw, t) for t in heads], 1)
    body = []
    for lab, tag, note in rows:
        a = load(PREFIX["scene"], tag, seed)
        sky = a[170:290, 150:450]
        m = sky.mean((0, 1))
        boost = np.clip((sky - m) * 6 + m, 0, 255)
        f = lambda x: cv2.resize(x, (cw, chh), interpolation=cv2.INTER_NEAREST)
        body.append(np.concatenate([txt(LW, chh, [lab, note]), f(sky), f(boost), f(a[700:820, 300:600])], 1))
    return [(f"Scene prompt, seed {seed}", np.concatenate([top, *body], 0))]


def main():
    global O
    args = sys.argv[1:]
    for flag, key in (("--dir", None), ("--grad-prefix", "grad"), ("--scene-prefix", "scene")):
        if flag in args:
            i = args.index(flag)
            value = args[i + 1]
            del args[i : i + 2]
            if key is None:
                O = value.rstrip("/") + "/"
            else:
                PREFIX[key] = value
    if not O:
        raise SystemExit(__doc__)
    out, title, kind = args[:3]
    rows = []
    for a in args[3:]:
        lab, rest = a.split("=", 1)
        tag, _, note = rest.partition("|")
        rows.append((lab, tag, note))
    blocks = grad(rows) if kind == "grad" else scene(rows)
    W = max(b.shape[1] for _, b in blocks)
    t = Image.new("RGB", (W, 30), "#222")
    ImageDraw.Draw(t).text((6, 6), title, fill="white", font=F)
    parts = [np.asarray(t)]
    for sub, b in blocks:
        s = Image.new("RGB", (W, 20), "#888")
        ImageDraw.Draw(s).text((6, 3), sub, fill="white", font=FS)
        parts += [np.asarray(s), np.pad(b, ((0, 0), (0, W - b.shape[1]), (0, 0)), constant_values=255)]
    Image.fromarray(np.concatenate(parts, 0).astype(np.uint8)).save(out)


main()
