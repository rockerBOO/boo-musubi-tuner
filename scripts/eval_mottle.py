"""Mottle evaluation. Compare image sets (same prompt, several seeds) per RGB channel, clipped regions excluded.

Usage (scipy, pywavelets and safetensors are not project dependencies):
  uv run --with scipy --with pywavelets --with safetensors python scripts/eval_mottle.py ref=GLOB other=GLOB ... [--latents ref=GLOB ...]

The first label is the reference; ratios are label / reference. Use a smooth prompt (the gradient prompt) and >= 3 seeds.

Metrics (all on unclipped pixels 0.06-0.94, eroded 8 px; mean over R,G,B unless noted):
  fine   SWT db2 level 1-2 detail rms x1e-4 (1-4 px)
  blob   SWT level 3 detail rms x1e-4 (~8 px, the blotch scale)
  rowH   level-2 H/V ratio (>1: horizontal-line bias)
  vax8   power at 8 px period on the vertical-frequency axis / neighbouring periods (periodic horizontal lines)
  lock   rms of the seed-averaged residual / mean per-image residual rms. Independent mottle gives ~1/sqrt(n).
         Near 1 means the same pattern in every seed (position-locked).
  clean  fraction of pixels used. Low values make the row unreliable.
  latHP  latent high-pass rms (x - gaussian 1.5 latent px), from --latents (safetensors from ComfyUI SaveLatent).
  flatHP same, only in the flattest 30% of the latent (use this for scene prompts); latstd = latent std (low = washed out).
  lap    Laplacian variance of luma x1e4: detail proxy that also rises with noise. Read it together with flatHP.
Headline metrics: blob and latHP. fine/rowH/vax8 are diagnostic only (validated: rowH and vax8 do NOT rank raw < turbo).
Controls to check the metric first: raw (no turbo) should read lower than turbo, and a VAE round trip lowest.
"""

import glob
import sys

import numpy as np
import pywt
from PIL import Image
from scipy import ndimage as ndi


def load(fn):
    im = np.asarray(Image.open(fn).convert("RGB")).astype(np.float64) / 255
    h, w = (im.shape[0] // 16) * 16, (im.shape[1] // 16) * 16
    return im[:h, :w]


def clean_mask(x, it=8):
    return ndi.binary_erosion((x > 0.06) & (x < 0.94), iterations=it)


def swt(x, m, levels=3):
    out = {}
    for lev, (_, (cH, cV, cD)) in zip(range(levels, 0, -1), pywt.swt2(x, "db2", level=levels)):
        mm = ndi.binary_erosion(m, iterations=2 ** (lev + 1))
        out[lev] = [float(np.sqrt((c[mm] ** 2).mean()) * 1e4) if mm.sum() > 5000 else np.nan for c in (cH, cV, cD)]
    return out


def vax8(x, m, T=128):
    res = x - ndi.gaussian_filter(x, 12)
    win = np.outer(np.hanning(T), np.hanning(T))
    P = []
    for y in range(0, x.shape[0] - T, T // 2):
        for xx in range(0, x.shape[1] - T, T // 2):
            if m[y : y + T, xx : xx + T].mean() < 0.98:
                continue
            t = res[y : y + T, xx : xx + T]
            P.append(np.abs(np.fft.fft2((t - t.mean()) * win)) ** 2)
    if not P:
        return np.nan
    P = np.mean(P, 0)
    fy, fx = np.fft.fftfreq(T)[:, None], np.fft.fftfreq(T)[None, :]
    r = np.hypot(fy, fx)
    ring = (np.abs(r - 1 / 8) < 0.012) & (np.abs(fx) < 0.01)
    nb = (np.abs(r - 1 / 8) >= 0.012) & (np.abs(r - 1 / 8) < 0.03)
    return float(P[ring].mean() / P[nb].mean())


def image_metrics(im):
    rows = []
    for c in range(3):
        x = im[..., c]
        m = clean_mask(x)
        s = swt(x, m)
        rows.append(
            {
                "fine": np.nanmean(s[1] + s[2]),
                "blob": np.nanmean(s[3]),
                "rowH": np.nanmean([s[2][0] / s[2][1]]),
                "vax8": vax8(x, m),
                "clean": float(m.mean()),
            }
        )
    return {k: [r[k] for r in rows] for k in rows[0]}  # metric -> [R, G, B]


def lock(images):
    """Position-locked share: rms of seed-averaged residual / mean per-image rms, per channel, same-size images only."""
    out = []
    for c in range(3):
        res, ms = [], []
        for im in images:
            x = im[..., c]
            ms.append(clean_mask(x))
            res.append(x - ndi.gaussian_filter(x, 12))
        m = np.all(ms, 0)
        if m.sum() < 5000 or len(res) < 2:
            out.append(np.nan)
            continue
        per = np.mean([r[m].std() for r in res])
        out.append(float(np.mean(res, 0)[m].std() / per))
    return out


def latent_stats(fn):
    """(whole-image latent HP, flat-region latent HP, latent std). Flat = lowest 30% blurred gradient (use for scenes)."""
    from safetensors import safe_open

    with safe_open(fn, "pt") as s:
        z = s.get_tensor("latent_tensor").float().numpy()[0, :, 0]
    whole = float(np.mean([(c - ndi.gaussian_filter(c, 1.5))[6:-6, 6:-6].std() for c in z]))
    g = ndi.gaussian_filter(z, (0, 1.5, 1.5))
    gm = np.mean([np.hypot(ndi.sobel(c, 0), ndi.sobel(c, 1)) for c in g], 0)
    m = ndi.binary_erosion(gm < np.percentile(gm, 30), iterations=2)
    m[:6] = m[-6:] = False
    m[:, :6] = m[:, -6:] = False
    flat = float(np.mean([(c - ndi.gaussian_filter(c, 1.5))[m].std() for c in z]))
    return whole, flat, float(z.std())


def summarize(files, latent_files):
    ims = [load(f) for f in files]
    by_size = {}
    for im in ims:
        by_size.setdefault(im.shape, []).append(im)
    per = [image_metrics(im) for im in ims]
    agg = {k: np.nanmean([np.nanmean(p[k]) for p in per]) for k in per[0]}
    chan = {k: np.nanmean([p[k] for p in per], 0) for k in ("fine", "blob")}
    lk = [np.nanmean(lock(v)) for v in by_size.values() if len(v) >= 2]
    agg["lock"] = float(np.nanmean(lk)) if lk else np.nan
    agg["n"] = len(ims)
    agg["exp_lock"] = float(np.mean([1 / np.sqrt(len(v)) for v in by_size.values() if len(v) >= 2])) if lk else np.nan
    if latent_files:
        ls = np.mean([latent_stats(f) for f in latent_files], 0)
        agg["latHP"], agg["flatHP"], agg["latstd"] = map(float, ls)
    agg["lap"] = float(np.mean([ndi.laplace(im.mean(2)).var() * 1e4 for im in ims]))
    return agg, chan


def parse(args):
    sets, lat = {}, {}
    cur = sets
    for a in args:
        if a == "--latents":
            cur = lat
            continue
        k, v = a.split("=", 1)
        cur[k] = sorted(glob.glob(v))
    return sets, lat


def main():
    sets, lat = parse(sys.argv[1:])
    ref = next(iter(sets))
    res = {k: summarize(f, lat.get(k, [])) for k, f in sets.items()}
    cols = ["fine", "blob", "rowH", "vax8", "lock", "clean"] + (["latHP", "flatHP", "latstd"] if lat else []) + ["lap"]
    print(f"{'label':28s}{'n':>3s} " + " ".join(f"{c:>8s}" for c in cols) + "   blob R/G/B (x1e-4)   exp.lock")
    for k, (a, ch) in res.items():
        print(
            f"{k:28s}{a['n']:3d} "
            + " ".join(f"{a.get(c, np.nan):8.3f}" for c in cols)
            + "   "
            + "/".join(f"{v:5.1f}" for v in ch["blob"])
            + f"   {a['exp_lock']:.2f}"
        )
    print(f"\nratio to {ref}:")
    r0 = res[ref][0]
    for k, (a, _) in res.items():
        print(
            f"{k:28s}    "
            + " ".join(f"{a.get(c, np.nan) / r0.get(c, np.nan):8.2f}" for c in cols if c not in ("clean", "lock", "latstd"))
        )


if __name__ == "__main__":
    main()
