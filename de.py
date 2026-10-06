#!/usr/bin/env python3
"""
layer7_verifier.py  --  Layer 7: Degradation-Aware Forensic Layer (DDV)
Author of the design: Prathmesh Kolte

DDV = Detect, Demultiplex, Verify.  A DEFENSIVE layer for an AI-image
verification platform.  It sits beside your five layers and does this:

  7.1  Degradation gate      : finds a pixel-block grid (size + offset)
  7.2  Depixelation          : demultiplexes the image back towards its source
                               (inverse of pixelate2's mixing; plain mosaics are
                               interpolated).  The restored copy is EXTRA evidence
  7.3  Demultiplexing        : unmixes the detail layer into Fourier / DCT /
                               Haar parts ("detail laid on top of blocks")
  7.5  Colour-linkage check  : real sensors keep R,G,B detail correlated
  7.6  Composite check       : patchy noise where a region was pasted in
  7.7  Fail-closed verdict   : veto rule instead of averaging scores

IMPORTANT
  * All thresholds are PLACEHOLDERS.  Calibrate them on your own labelled
    real / AI / degraded images before trusting any result.
  * No layer proves authenticity.  Only a valid C2PA chain can lead to
    "verified_authentic".  "no_objection" is NOT proof.
  * Restoration is an estimate (pixelation is many-to-one).  The restored
    copy is extra evidence, never a replacement for the original.

USAGE
  python layer7_verifier.py --selftest
  python layer7_verifier.py photo.jpg
  python layer7_verifier.py folder/ --json report.json
  python layer7_verifier.py photo.png \
        --layers '{"neural":0.95,"fingerprint":0.95,"exif":0.10,"chroma":0.10}' \
        --c2pa absent --restored-out restored/

  --layers : probabilities of "real" (0..1) from YOUR platform's layers
             (keys: neural, fingerprint, exif, chroma).  Optional.
  --c2pa   : result of your C2PA layer: valid | invalid | absent (default).
Requires: numpy, scipy, pillow
"""
import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn, fft2, ifft2, irfft2, rfft2
from scipy.ndimage import correlate1d, gaussian_filter, uniform_filter1d

# --------------------------------------------------------------------------
# Placeholder thresholds -- CALIBRATE THESE ON REAL DATA
# --------------------------------------------------------------------------
DEFAULT_THR = dict(
    block=1.25,     # grid strength above this  => degraded (blocks found) - lowered for screenshots
    flat=0.05,      # share of flat tiles above this (needs some grid support) - lowered for screenshots
    link=0.45,      # colour linkage below this => weak_colour_linkage
    patchy=0.12,    # noise patchiness above this => possible composite
    r2=0.15,        # demux R^2 above this (on degraded image) => detail_on_blocks
    veto=0.30,      # a hard-evidence layer below this vetoes "real"
    max_side=1536,  # analyse at most this many pixels per side (centre crop)
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def to_gray(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def highpass(ch, sigma=1.5):
    return ch - gaussian_filter(ch, sigma)


def exif_summary(img):
    """Very small EXIF summary (layer 3 remains YOUR platform's job)."""
    software = img.info.get("Software", "")
    try:
        ex = img.getexif()
        software = software or ex.get(305, "")
        make, model, dt = ex.get(271), ex.get(272), ex.get(306)
        return dict(present=bool(ex), make=make, model=model, datetime=dt, software=software)
    except Exception:
        return dict(present=False, make=None, model=None, datetime=None, software=software)


def load_image(path, max_side):
    img = Image.open(path)
    exif = exif_summary(img)
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    note = None
    if max(h, w) > max_side:                 # centre crop keeps the pixel grid intact
        ch, cw = min(h, max_side), min(w, max_side)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        rgb = rgb[y0:y0 + ch, x0:x0 + cw]
        note = f"analysed centre crop {cw}x{ch} of {w}x{h}"
    return np.ascontiguousarray(rgb), exif, note


# --------------------------------------------------------------------------
# 7.1  Degradation gate: block size AND grid offset
# --------------------------------------------------------------------------
def _best_phase(prof, b):
    """Best boundary phase for period b. Returns (strength, first_block_pixel)."""
    m = prof.mean() + 1e-9
    best, best_r = 0.0, 0
    for r in range(b):                       # diff index r lies between pixel r and r+1
        v = prof[r::b].mean() / m
        if v > best:
            best, best_r = v, r
    return best, (best_r + 1) % b


def estimate_grid(gray, bmin=2, bmax=64):
    """Return dict(b, strength, px, py).  strength ~1 means 'no grid'."""
    g = gaussian_filter(gray, 1.2)           # suppress grain, keep block steps
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)
    bmax = max(bmin, min(bmax, min(gray.shape) // 8))   # need >= 8 blocks per axis
    res = {}
    for b in range(bmin, bmax + 1):
        sx, px = _best_phase(dx, b)
        sy, py = _best_phase(dy, b)
        res[b] = (min(sx, sy), px, py)
    top = max(v[0] for v in res.values())
    # multiples of the true size also score high -> take the SMALLEST strong period
    # strength is relative to 1.0 (baseline), so compare the peak height above 1.0
    b = min(k for k, v in res.items() if (v[0] - 1.0) >= 0.7 * (top - 1.0))
    s, px, py = res[b]
    return dict(b=b, strength=float(s), px=px, py=py)


def flat_tile_ratio(gray, b, eps=8.0):
    if b < 2:
        return 0.0
    h, w = (gray.shape[0] // b) * b, (gray.shape[1] // b) * b
    if h == 0 or w == 0:
        return 0.0
    t = gray[:h, :w].reshape(h // b, b, w // b, b).std(axis=(1, 3))
    return float((t < eps).mean())


def palette_colors(rgb):
    """Distinct 5-bit colours (reported only; not used until calibrated)."""
    q = (rgb >> 3).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    return int(len(np.unique(key)))


# --------------------------------------------------------------------------
# 7.2  Depixelation = demultiplexing.   pixelate2 builds its output as
#
#          Y = S x  +  alpha * F x  +  beta * D x  +  gamma * W x        (x = source)
#
#      S  nearest-neighbour block base       F  Fourier high-pass of blur(x)
#      D  DCT residual of blur(x)            W  boosted Haar detail of blur(x)
#
#      and then clips, unsharp-masks and palette-quantises it.  Every branch is
#      LINEAR in x, so depixelating is solving that system for x (robust,
#      regularised least squares), not smoothing blocks and hoping.
#
#      What cannot come back: whatever the two palette quantisations and the
#      clipping threw away.  The result is close to the source, never identical.
# --------------------------------------------------------------------------
MUX_PRESETS = {     # block size -> what pixelate2's preset injected (+ solver weights tuned on real photos)
    4: dict(name="clear",  alpha=0.35, beta=0.20, gamma=0.15, clarity=2.0, omega=0.03, lam=0.02),
    6: dict(name="poster", alpha=0.30, beta=0.15, gamma=0.10, clarity=2.5, omega=0.03, lam=0.02),
    8: dict(name="retro",  alpha=0.40, beta=0.25, gamma=0.20, clarity=1.5, omega=0.03, lam=0.04),
}
_F32 = np.float32
_LUMA = np.array([19595, 38470, 7471]) / 65536.0          # PIL's "L" weights (used by pixelate2.prep)
_OPP = np.array([[1, 1, 1], [1, 0, -1], [1, -2, 1]], dtype=np.float64)
_OPP = (_OPP / np.linalg.norm(_OPP, axis=1, keepdims=True)).astype(_F32)   # luma + 2 chroma, orthonormal


def _pil_taps(radius):
    """One of the 3 box passes PIL's GaussianBlur(radius) is made of, as a 1-D kernel."""
    s2 = radius * radius / 3.0
    l = int(np.floor((np.sqrt(12.0 * s2 + 1.0) - 1.0) / 2.0))
    a = (2 * l + 1) * (l * (l + 1) - 3.0 * s2) / (6.0 * (s2 - (l + 1) ** 2))
    k = np.ones(2 * l + 3)
    k[0] = k[-1] = a
    return (k / k.sum()).astype(_F32)


def _unsharp(x, pct, radius=1.2, thr=12.0):
    """PIL's UnsharpMask on a float (C, H, W) array (pixelate2's last filter)."""
    blur, k = x, _pil_taps(radius)
    for ax in (2, 1):
        for _ in range(3):
            blur = correlate1d(blur, k, axis=ax, mode="nearest")
    d = x - blur
    return np.clip(np.where(np.abs(d) > thr, x + d * pct, x), 0, 255)


def _unsharp_inverse(y, pct, iters=6):
    """Undo _unsharp by damped fixed-point iteration (its Jacobian lies in [1, 1 + pct])."""
    tau = 2.0 / (2.0 + pct)
    x = y.copy()
    for _ in range(iters):
        x += tau * (y - _unsharp(x, pct))
    return x


def _pil_maps(n, b):
    """Block layout PIL's NEAREST down(n -> n//b) + up(-> n) really produces, read off PIL itself.
    Returns (lab, src): lab[i] = block of pixel i, src[k] = the source pixel block k shows."""
    m = max(1, n // b)
    idx = Image.fromarray(np.arange(n, dtype=np.int32)[None, :], mode="I")
    src = np.asarray(idx.resize((m, 1), Image.NEAREST)).ravel().astype(np.int64)
    blk = Image.fromarray(np.arange(m, dtype=np.int32)[None, :], mode="I")
    lab = np.asarray(blk.resize((n, 1), Image.NEAREST)).ravel().astype(np.int64)
    return lab, src


def _even_maps(n, b):
    """Same, for a canvas whose blocks are exactly b wide and start at 0."""
    return np.arange(n) // b, np.minimum(np.arange(-(-n // b)) * b + b // 2, n - 1)


class _Mux:
    """The multiplexer as a linear operator on float32 (C, H, W) arrays:  Y = S x + K G x."""

    def __init__(self, h, w, b, pr, maps=None, blur=0.75, fsigma=15.0, keep=0.15):
        self.h, self.w, self.b = h, w, b
        self.al, self.be, self.g2 = _F32(pr["alpha"]), _F32(pr["beta"]), _F32(2.0 * pr["gamma"])
        self.a = _F32(pr["alpha"] + pr["beta"] + 2.0 * pr["gamma"])
        (self.rlab, self.rsrc), (self.clab, self.csrc) = maps or (_pil_maps(h, b), _pil_maps(w, b))
        self.rstart = np.flatnonzero(np.diff(self.rlab, prepend=-1))
        self.cstart = np.flatnonzero(np.diff(self.clab, prepend=-1))
        rc, cc = np.diff(np.append(self.rstart, h)), np.diff(np.append(self.cstart, w))
        self.icnt = (1.0 / (rc[:, None] * cc[None, :])).astype(_F32)
        self.rsel, self.csel = self.rsrc[self.rlab], self.csrc[self.clab]
        self.fy, self.fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
        k = _pil_taps(blur).astype(np.float64)
        l = len(k) // 2
        box = lambda f: sum(k[l + j] * np.cos(2 * np.pi * j * f) for j in range(-l, l + 1))
        self.gf = ((box(self.fy) * box(self.fx)) ** 3).astype(_F32)                 # the 0.75 px pre-blur
        self.lf = np.exp(-((self.fy * h) ** 2 + (self.fx * w) ** 2) / (2.0 * fsigma ** 2)).astype(_F32)
        self.glf = (self.gf * self.lf * self.al).astype(_F32)                       # Fourier low-pass
        self.rk, self.ck = max(1, int(h * keep)), max(1, int(w * keep))             # DCT keep window
        self.he, self.we = h - h % 2, w - w % 2                                     # Haar works on even size

    def fft(self, x):
        return rfft2(x, axes=(1, 2), workers=-1)

    def ifft(self, X):
        return irfft2(X, s=(self.h, self.w), axes=(1, 2), workers=-1)

    def dct_low(self, x):
        C = dctn(x, axes=(1, 2), norm="ortho", workers=-1)
        C[:, self.rk:] = 0
        C[:, :, self.ck:] = 0
        return idctn(C, axes=(1, 2), norm="ortho", workers=-1, overwrite_x=True)

    def haar_low(self, x):
        he, we = self.he, self.we
        m = (x[:, 0:he:2, 0:we:2] + x[:, 0:he:2, 1:we:2] + x[:, 1:he:2, 0:we:2] + x[:, 1:he:2, 1:we:2]) * _F32(0.25)
        out = x.copy()
        out[:, 0:he:2, 0:we:2] = m
        out[:, 0:he:2, 1:we:2] = m
        out[:, 1:he:2, 0:we:2] = m
        out[:, 1:he:2, 1:we:2] = m
        return out

    def bsum(self, x):
        return np.add.reduceat(np.add.reduceat(x, self.rstart, axis=1), self.cstart, axis=2)

    def bmean(self, x):
        return self.bsum(x) * self.icnt

    def up(self, blk):
        return blk[:, self.rlab][:, :, self.clab]

    def A(self, x):                                   # projector onto block-constant images
        return self.up(self.bmean(x))

    def S(self, x):                                   # the pixelation itself (sample + hold)
        return x[:, self.rsel][:, :, self.csel]

    def St(self, y):
        out = np.zeros_like(y)
        out[:, self.rsrc[:, None], self.csrc[None, :]] = self.bsum(y)
        return out

    def _k0(self, v):                                 # DCT + Haar branches (+ the identity parts)
        return self.a * v - self.be * self.dct_low(v) - self.g2 * self.haar_low(v)

    def KG(self, x, X=None):                          # the whole detail layer
        X = self.fft(x) if X is None else X
        return self._k0(self.ifft(X * self.gf)) - self.ifft(X * self.glf)

    def GK(self, y, extra=None):                      # its adjoint
        Z = self.fft(self._k0(y)) * self.gf - self.fft(y) * self.glf
        return self.ifft(Z if extra is None else Z + extra)

    def gain(self):
        """Fourier-diagonal approximation of K G (preconditioner only)."""
        a2 = (np.cos(np.pi * self.fy) ** 2) * (np.cos(np.pi * self.fx) ** 2)
        ld = (np.abs(self.fy) * 2 < self.rk / self.h) & (np.abs(self.fx) * 2 < self.ck / self.w)
        return ((self.a - self.al * self.lf - self.be * ld - self.g2 * a2) * self.gf).astype(_F32)


def _unmix(canvas, m, pr, mask=None, blur=0.0, outer=3, inner=6, kout=3.0):
    """Solve  Y = S x + K G x  for x.   canvas: H x W x 3 on the model's grid.  Returns float32 H x W x 3.

    Weighted least squares: the in-block detail equations are trusted fully, the block-base
    equations by `omega` (the base went through a second, coarser palette), plus a smoothness
    prior (~ f^3, the inverse of a natural-image spectrum) that is stronger on chroma.
    Clipping, outline pixels and unsharp-mask leftovers are handled by re-linearising around
    the current estimate and rejecting outlying residuals.
    `blur`: Gaussian blur (px) a screenshot / re-encode put on top of the mix; modelled, not ignored."""
    h, w, b = m.h, m.w, m.b
    X = np.ascontiguousarray(np.moveaxis(canvas.astype(_F32), 2, 0))
    X = _unsharp_inverse(X, int(60 * pr["clarity"]) / 100.0)
    hb = np.exp(-2 * np.pi ** 2 * blur ** 2 * (m.fy ** 2 + m.fx ** 2)).astype(_F32) if blur > 0 else None
    soft = (lambda v: m.ifft(m.fft(v) * hb)) if blur > 0 else (lambda v: v)    # blur the capture added on top
    omega, om1 = pr["omega"], _F32(pr["omega"] - 1.0)
    T = m.gain()
    rough = (4 * np.sin(np.pi * m.fy) ** 2 + 4 * np.sin(np.pi * m.fx) ** 2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = lambda f: np.where(np.abs(f) < 1e-9, 1.0, np.sin(np.pi * f * b) / (b * np.sin(np.pi * f)))
        DA = sinc(m.fy) ** 2 * sinc(m.fx) ** 2
    lamr = (np.array([1.0, 10.0, 10.0]).reshape(3, 1, 1) * pr["lam"] * rough[None]).astype(_F32)
    h2 = 1.0 if hb is None else hb[None] ** 2
    Mi = (1.0 / ((T[None] ** 2 * (1 - (1 - omega) * DA[None]) + omega) * h2 + lamr + 1e-6)).astype(_F32)
    opp = lambda v: np.tensordot(_OPP, v, axes=(1, 0))
    oppT = lambda v: np.tensordot(_OPP.T, v, axes=(1, 0))

    def normal(x):                                    # F^T C F + prior
        Xf = m.fft(x)
        v = soft(m.S(x) + m.KG(x, Xf))
        v = soft(v + om1 * m.A(v))
        return m.St(v) + m.GK(v, Xf * lamr)

    def rhs(v):
        v = soft(v + om1 * m.A(v))
        return m.St(v) + m.GK(v)

    def pcg(bvec, x, n):
        r = bvec - normal(x)
        z = m.ifft(m.fft(r) * Mi)
        p = z.copy()
        rz = (r * z).sum(axis=(1, 2), keepdims=True)
        for _ in range(n):
            Hp = normal(p)
            a = rz / ((p * Hp).sum(axis=(1, 2), keepdims=True) + 1e-20)
            x += a * p
            r -= a * Hp
            z = m.ifft(m.fft(r) * Mi)
            rz2 = (r * z).sum(axis=(1, 2), keepdims=True)
            p = z + (rz2 / (rz + 1e-20)) * p
            rz = rz2
        return x

    def tame(res, hard):
        """Split the residual into a block-constant part (base error) and a pixel part; drop outliers."""
        wgt = np.ones_like(res) if mask is None else np.broadcast_to(mask, res.shape).astype(_F32)
        for _ in range(2):
            rA = m.up(m.bsum(res * wgt) / np.maximum(m.bsum(wgt), 1e-6))
            rP = res - rA
            sig = 1.4826 * np.median(np.abs(rP[:, ::3, ::3])) + 0.5
            wgt = (np.abs(rP) < kout * sig).astype(_F32)
            if mask is not None:
                wgt *= mask
        rP = rP * wgt if hard else np.clip(rP, -kout * sig, kout * sig) * (1.0 if mask is None else mask)
        return rA + rP

    bm = m.bmean(X)
    x = np.stack([np.asarray(Image.fromarray(bm[c]).resize((w, h), Image.BICUBIC)) for c in range(3)]).astype(_F32)
    for o in range(outer):
        pred = soft(m.S(x) + m.KG(x))
        res = tame(X - np.clip(pred, 0, 255), hard=(o >= outer - 2))
        x = oppT(pcg(rhs(opp(pred + res)), opp(x), inner))
    return np.clip(np.moveaxis(x, 0, 2), 0, 255)


def _undo_prep(x, clarity):
    """Undo pixelate2.prep's colour and contrast boosts (its autocontrast cannot be undone blindly)."""
    x = x.astype(np.float64)
    g = (x @ _LUMA)[..., None]
    x = g + (x - g) / (1 + 0.15 * clarity)
    mean = np.floor((x @ _LUMA).mean() + 0.5)
    return np.clip(mean + (x - mean) / (1 + 0.08 * clarity), 0, 255)


# ---- locating the block grid with sub-pixel precision (screenshots / chat apps rescale it) ----
def _grid_axis(prof, pmin=2.6, pmax=40.0, pad=16):
    """Period, phase and peak/median SNR of the block comb in a mean-|gradient| profile.
    Block boundaries sit at  off + k * period  (pixel i covers [i, i + 1))."""
    n = len(prof)
    q = prof - uniform_filter1d(prof, 31, mode="nearest")
    N = pad * (1 << int(np.ceil(np.log2(n))))
    S = np.abs(np.fft.rfft(q * np.hanning(n), N))
    lo, hi = int(np.ceil(N / pmax)), int(N / pmin)
    k = lo + int(np.argmax(S[lo:hi]))
    for div in (2, 3):                                   # a strong sub-harmonic is the real block size
        j = int(round(k / div))
        if j - 2 * pad >= lo:
            j = j - 2 * pad + int(np.argmax(S[j - 2 * pad:j + 2 * pad + 1]))
            if S[j] > 0.6 * S[k]:
                k = j
                break
    d = 0.5 * (S[k - 1] - S[k + 1]) / (S[k - 1] - 2 * S[k] + S[k + 1] + 1e-12)
    fk = (k + d) / N
    p = 1.0 / fk
    x = np.arange(n)
    x0 = (-np.angle(np.sum(q * np.exp(-2j * np.pi * fk * x))) / (2 * np.pi * fk)) % p
    best, bo = -1e9, x0
    for dlt in np.linspace(-0.5, 0.5, 21):               # refine the phase on the profile itself
        v = np.interp(np.arange(x0 + dlt, n - 1, p), x, prof).mean()
        if v > best:
            best, bo = v, x0 + dlt
    return p, (bo + 1.0) % p, float(S[k] / (np.median(S[lo:hi]) + 1e-9))


def _comb(prof, pos):
    """Mean gradient at boundary coordinates `pos`, relative to the profile mean (1 = no grid there)."""
    pos = np.asarray(pos, dtype=np.float64) - 1.0
    pos = pos[(pos >= 0) & (pos <= len(prof) - 1)]
    if len(pos) < 4:
        return 0.0
    return float(np.interp(pos, np.arange(len(prof)), prof).mean() / (prof.mean() + 1e-9))


def _axis_grid(prof, n):
    """One axis: is the grid pixelate2's own layout ('pil'), an integer grid ('int') or rescaled ('scaled')?"""
    p, off, snr = _grid_axis(prof)
    score = _comb(prof, np.arange(off, n, p))
    out = dict(kind="scaled", p=float(p), off=float(off), b=None,
               ok=snr >= 12 * np.sqrt(n / 1000.0))       # the comb's SNR grows like sqrt(n)
    bi = int(round(p))
    if bi >= 2 and abs(p - bi) <= 0.03 * bi:
        s_pil = _comb(prof, np.flatnonzero(np.diff(_pil_maps(n, bi)[0])) + 1)
        s_int, o_int = max((_comb(prof, np.arange(o if o else bi, n, bi)), o) for o in range(bi))
        if s_pil >= 0.985 * max(s_int, score):           # on an exact grid the boundary gradient itself decides
            out.update(kind="pil", b=bi, off=0.0, p=n / max(1, n // bi), ok=s_pil >= 1.5)
        elif s_int >= 0.97 * score:
            out.update(kind="int", b=bi, off=float(o_int), p=float(bi), ok=s_int >= 1.5)
    return out


def locate_grid(gray):
    """Sub-pixel block grid of a (possibly cropped / rescaled) pixelated image, or None."""
    h, w = gray.shape
    if min(h, w) < 64:
        return None
    gx = _axis_grid(np.abs(np.diff(gray, axis=1)).mean(axis=0), w)
    gy = _axis_grid(np.abs(np.diff(gray, axis=0)).mean(axis=1), h)
    if not (gx["ok"] and gy["ok"]) or abs(gx["p"] - gy["p"]) > 0.04 * max(gx["p"], gy["p"]):
        return None
    return gx, gy


def _to_canvas(rgb, gx, gy):
    """Bring the image onto a canvas whose blocks are whole pixels starting at (0, 0).
    Returns (canvas, mask, maps, b, scale, kind, back); `back` maps a canvas image to the input frame."""
    h, w = rgb.shape[:2]
    exact = ("pil", "int")
    if gx["kind"] == gy["kind"] == "pil" and gx["b"] == gy["b"]:              # untouched pixelate2 output
        return rgb, None, None, gx["b"], 1.0, "native", lambda o: o
    if gx["kind"] in exact and gy["kind"] in exact and gx["b"] == gy["b"]:    # cropped: pad to the grid
        b = gx["b"]
        pl, pt = (b - int(round(gx["off"]))) % b, (b - int(round(gy["off"]))) % b
        canvas = np.pad(rgb, ((pt, (-(h + pt)) % b), (pl, (-(w + pl)) % b), (0, 0)), mode="edge")
        mask = np.zeros(canvas.shape[:2], _F32)
        mask[pt:pt + h, pl:pl + w] = 1
        maps = (_even_maps(canvas.shape[0], b), _even_maps(canvas.shape[1], b))
        return canvas, mask, maps, b, 1.0, "cropped", lambda o: o[pt:pt + h, pl:pl + w]
    p = 0.5 * (gx["p"] + gy["p"])                                             # rescaled: resample to the grid
    b = 4 if p < 5.9 else (6 if p < 6.93 else 8)       # nearest preset, leaning to "clear" (the default,
                                                       # and the one a wrong guess hurts most)
    zx, zy = gx["p"] / b, gy["p"] / b
    P = int(np.ceil(max(gx["p"], gy["p"]))) + 2
    src = Image.fromarray(np.pad(rgb, ((P, P), (P, P), (0, 0)), mode="edge"))
    seen = np.zeros((h + 2 * P, w + 2 * P), np.uint8)
    seen[P:P + h, P:P + w] = 255
    x0 = gx["off"] + P - gx["p"] * np.floor((gx["off"] + P - 1) / gx["p"])
    y0 = gy["off"] + P - gy["p"] * np.floor((gy["off"] + P - 1) / gy["p"])
    nbx = int(np.floor((w + 2 * P - 1 - x0) / gx["p"]))
    nby = int(np.floor((h + 2 * P - 1 - y0) / gy["p"]))
    box = (x0, y0, x0 + nbx * gx["p"], y0 + nby * gy["p"])
    size = (nbx * b, nby * b)
    canvas = np.asarray(src.resize(size, Image.LANCZOS if p > b else Image.BICUBIC, box=box))
    mask = (np.asarray(Image.fromarray(seen).resize(size, Image.BILINEAR, box=box)) > 250).astype(_F32)
    maps = (_even_maps(size[1], b), _even_maps(size[0], b))
    bbox = ((P - x0) / zx, (P - y0) / zy, (P + w - x0) / zx, (P + h - y0) / zy)

    def back(o):
        img = Image.fromarray(np.round(np.clip(o, 0, 255)).astype(np.uint8))
        return np.asarray(img.resize((w, h), Image.BICUBIC, box=bbox))
    return canvas, mask, maps, b, float(p / b), "rescaled", back


def _capture_blur(canvas, m):
    """Gaussian blur (px) the block edges picked up AFTER pixelation (screenshots, re-encodes), read
    from how far the boundary gradient has spread into the blocks.  None = too smeared to undo.
    Knots calibrated on pixelate2 outputs blurred by known amounts (b = 8 uses the second
    neighbour because the retro preset's outline already widens the first one)."""
    g, b, ph = to_gray(canvas.astype(np.float64)), m.b, 0.0
    for ax, lab, start in ((1, m.clab, m.cstart), (0, m.rlab, m.rstart)):
        prof = np.abs(np.diff(g, axis=ax)).mean(axis=1 - ax)
        pos = np.arange(1, len(lab)) - start[lab[1:]]                     # 0 = the step into a new block
        ph = ph + np.array([prof[pos == k].mean() for k in range(b)])     # [boundary, +1, +2, ...]
    d = 2 if b >= 7 else 1
    r = (0.5 * (ph[d] + ph[-d]) - ph[b // 2]) / (ph[0] - ph[b // 2] + 1e-9)
    knots = (0.06, 0.13, 0.22, 0.40) if b >= 7 else (0.17, 0.26, 0.50, 0.76 if b >= 6 else 0.58)
    if r > knots[-1] + 0.12:
        return None
    return float(np.interp(r, knots, (0.0, 0.6, 1.0, 1.5)))


def _plain_evidence(canvas, m, scale=None):
    """Could the in-block residual be nothing but JPEG / resampling damage to a PLAIN block mosaic?
    Returns (flat, corr): median in-block std, and the best correlation between the observed
    residual and the residual a plain mosaic shows after that same damage (simulated)."""
    X = np.moveaxis(canvas.astype(_F32), 2, 0)
    mos = m.up(m.bmean(X))
    lw = np.array([0.299, 0.587, 0.114], dtype=_F32)
    R = np.tensordot(lw, X - mos, axes=(0, 0))
    flat = float(np.median(np.sqrt(m.bmean((R * R)[None])[0])))
    h, w = R.shape
    y0, x0 = max(0, ((h - 768) // 2) // 24 * 24), max(0, ((w - 768) // 2) // 24 * 24)   # keeps JPEG + block grids
    win = (slice(y0, min(h, y0 + 768)), slice(x0, min(w, x0 + 768)))
    Ro = R[win]
    Mo = np.round(np.clip(np.moveaxis(mos, 0, 2)[win], 0, 255)).astype(np.uint8)
    Ml = Mo.astype(_F32) @ lw
    sims = []
    if scale is None:                                    # native grid: what would JPEG do to the mosaic?
        for q in (30, 40, 50, 60, 68, 75, 82, 88, 93):
            buf = io.BytesIO()
            Image.fromarray(Mo).save(buf, "JPEG", quality=q)
            sims.append(np.asarray(Image.open(buf).convert("RGB")))
    else:                                                # rescaled grid: what would resizing do to it?
        hh, ww = Mo.shape[:2]
        for flt in (Image.BILINEAR, Image.BICUBIC):
            big = Image.fromarray(Mo).resize((max(1, round(ww * scale)), max(1, round(hh * scale))), flt)
            sims.append(np.asarray(big.resize((ww, hh), Image.LANCZOS if scale > 1 else Image.BICUBIC)))
    corr = 0.0
    for s in sims:
        Rs = s.astype(_F32) @ lw - Ml
        corr = max(corr, float((Ro * Rs).sum() / (np.sqrt((Ro * Ro).sum() * (Rs * Rs).sum()) + 1e-9)))
    return flat, corr


def _mark_synthetic(img):
    """Invisible +-0.8 % two-pixel checkerboard put on every reconstruction (kept from the previous
    restore()): it trips neural AI-image detectors, so a restored copy cannot pass as a camera original."""
    k = np.full(img.shape[:2], 0.992, _F32)
    k[0::2, 0::2] = k[1::2, 1::2] = 1.008
    return np.clip(np.round(img * k[..., None]), 0, 255).astype(np.uint8)


def restore(rgb, b=None, dm=None, info=None, **_):
    """Depixelate `rgb` (H x W x 3 uint8).  Returns a uint8 image of the same size.

    1. locate the block grid with sub-pixel precision (survives crops, screenshots, chat apps),
    2. bring the image onto a canvas where the grid is whole pixels (pad or resample),
    3. plain block mosaic  -> smooth interpolation of the block colours (nothing else is in there),
       pixelate2 multiplex -> demultiplex: solve the mixing equations for the source image,
       then undo pixelate2's colour / contrast pre-boost,
    4. map the result back onto the input frame.

    `b`, `dm` and any other keyword (the degradation gate's integer guess, its regression, older
    call signatures) are accepted for backward compatibility only: the grid is re-located here.
    `info`, if given, is filled with what was found and done."""
    info = {} if info is None else info
    info.update(method="none", block=None, scale=None, preset=None)
    grid = locate_grid(to_gray(rgb.astype(np.float64)))
    if grid is None:                                     # no block grid to undo: leave the pixels alone
        return rgb.copy()
    canvas, mask, maps, cb, scale, kind, back = _to_canvas(rgb, *grid)
    pr = MUX_PRESETS[min(MUX_PRESETS, key=lambda c: abs(c - cb))]
    m = _Mux(canvas.shape[0], canvas.shape[1], cb, pr, maps=maps)
    flat, corr = _plain_evidence(canvas, m, scale if kind == "rescaled" else None)
    info.update(block=int(cb), scale=round(scale, 4), grid=kind,
                in_block_std=round(flat, 2), plain_corr=round(corr, 2))
    if kind == "rescaled":
        plain = flat < 1.0 or (corr > 0.6 and flat < 1.8)
    else:
        plain = flat < 1.0 or corr > 0.43 or (corr > 0.25 and flat < 1.5)
    if plain:
        bm = m.bmean(np.moveaxis(canvas.astype(_F32), 2, 0))
        size = (canvas.shape[1], canvas.shape[0])
        out = np.stack([np.asarray(Image.fromarray(c).resize(size, Image.BICUBIC)) for c in bm], axis=-1)
        info.update(method="interpolate")
    else:
        blur = _capture_blur(canvas, m)
        if blur is None:                                 # a grid, but no block edges left to work from
            info.update(note="block edges too blurred to demultiplex")
            return rgb.copy()
        if kind == "rescaled" and abs(scale - 1.0) > 0.01:
            blur = float(np.hypot(blur, 0.5))            # resampling alone costs ~0.5 px
        elif blur < 0.6:
            blur = 0.0                                   # untouched grid: trust the exact model
        out = _unmix(canvas, m, pr, None if mask is None else mask[None], blur=blur)
        out = _undo_prep(out, pr["clarity"])
        info.update(method="demultiplex", preset=pr["name"], alpha=pr["alpha"], beta=pr["beta"], gamma=pr["gamma"],
                    capture_blur=round(blur, 2))
    out = back(np.round(np.clip(out, 0, 255)).astype(np.uint8))
    return np.ascontiguousarray(_mark_synthetic(out))


# --------------------------------------------------------------------------
# 7.3  Demultiplexing: unmix the detail layer into Fourier / DCT / Haar parts
# --------------------------------------------------------------------------
def domain_maps(gray):
    """Fourier high-pass E, DCT residual D, Haar detail W of the observed image."""
    h, w = gray.shape
    F = fft2(gray)
    u = np.fft.fftfreq(h)[:, None]
    v = np.fft.fftfreq(w)[None, :]
    H = np.exp(-(u ** 2 + v ** 2) * (2 * np.pi ** 2) * 1.5 ** 2)   # Gaussian low-pass
    E = gray - np.real(ifft2(F * H))
    C = dctn(gray, norm="ortho")
    M = np.zeros_like(C)
    k = int(np.sqrt(0.15) * min(C.shape))                          # keep lowest ~15 %
    M[:k, :k] = 1
    D = gray - idctn(C * M, norm="ortho")
    hh, ww = (h // 2) * 2, (w // 2) * 2
    g = gray[:hh, :ww]
    LL = g.reshape(hh // 2, 2, ww // 2, 2).mean(axis=(1, 3))
    W = np.zeros_like(gray)
    W[:hh, :ww] = g - np.kron(LL, np.ones((2, 2)))                 # Haar detail bands
    return E, D, W

def domain_maps_rgb(rgb):
    E, D, W = [], [], []
    for i in range(3):
        e, d, w = domain_maps(rgb[:,:,i].astype(float))
        E.append(e); D.append(d); W.append(w)
    return np.stack(E, axis=-1), np.stack(D, axis=-1), np.stack(W, axis=-1)

def demultiplex(rgb, b):
    """Model I = Base + R (Base = block means). Regress R on E, D, W.
    High R^2 on a blocky image => detail was laid on top of blocks."""
    empty = {"r2": 0.0, "coefs": (0.0, 0.0, 0.0), "cross_corr": 0.0}
    if b < 2:
        return empty
    gray = to_gray(rgb.astype(float))
    h, w = (gray.shape[0] // b) * b, (gray.shape[1] // b) * b
    if h < 16 or w < 16:
        return empty
    g = gray[:h, :w]
    base = np.kron(g.reshape(h // b, b, w // b, b).mean(axis=(1, 3)), np.ones((b, b)))
    R = (g - base).ravel()
    if R.std() < 1e-3:                       # clean pixelation: no detail layer at all
        return empty
    E, D, W = [m[:h, :w].ravel() for m in domain_maps(gray)]
    A = np.stack([E, D, W], axis=1)
    coefs, *_ = np.linalg.lstsq(A, R, rcond=None)
    pred = A @ coefs
    r2 = 1 - ((R - pred) ** 2).sum() / (((R - R.mean()) ** 2).sum() + 1e-9)
    cc = np.nan_to_num(np.corrcoef(A.T))
    cross = float(np.mean([abs(cc[0, 1]), abs(cc[0, 2]), abs(cc[1, 2])]))
    return {"r2": float(r2), "coefs": tuple(float(c) for c in coefs), "cross_corr": cross}


# --------------------------------------------------------------------------
# 7.5  Colour linkage (sensor-like)      7.6  Composite / patchy noise
# --------------------------------------------------------------------------
def channel_linkage(rgb):
    f = rgb.astype(float)
    hp = [highpass(f[..., i]).ravel() for i in range(3)]
    c = np.nan_to_num(np.corrcoef(hp))       # flat channels give NaN -> 0
    return float(np.mean([c[0, 1], c[0, 2], c[1, 2]]))


def noise_patchiness(gray, tile=32):
    hp = highpass(gray, 1.0)
    h, w = (hp.shape[0] // tile) * tile, (hp.shape[1] // tile) * tile
    if h == 0 or w == 0:
        return 0.0
    t = hp[:h, :w].reshape(h // tile, tile, w // tile, tile).std(axis=(1, 3))
    return float(t.std() / (t.mean() + 1e-9))   # high = patchy noise


# --------------------------------------------------------------------------
# Layer 7 analysis
# --------------------------------------------------------------------------
def analyse(rgb, thr=None, exif=None, full=None):
    thr = {**DEFAULT_THR, **(thr or {})}
    gray = to_gray(rgb.astype(float))
    grid = estimate_grid(gray)
    b, strength = grid["b"], grid["strength"]

    rgb_c = rgb[grid["py"]:, grid["px"]:]                # align to the block grid
    gray_c = gray[grid["py"]:, grid["px"]:]
    flat = flat_tile_ratio(gray_c, b)
    degraded = strength > thr["block"] or (b >= 2 and strength > 1.15 and flat > thr["flat"]) or (flat > 0.20)

    dm = demultiplex(rgb_c, b) if degraded else {"r2": 0.0, "coefs": (0, 0, 0), "cross_corr": 0.0}
    restored, rinfo = None, {}
    if degraded:
        try:                                             # `full`: the uncropped frame, when analysis used a crop
            restored = restore(rgb if full is None else full, b, dm, rinfo)
        except Exception as e:                           # restoration is extra evidence; keep the verdict
            rinfo = dict(method="failed", error=str(e))
    link = channel_linkage(rgb)
    patchy = 0.0 if degraded else noise_patchiness(gray)  # noise stats meaningless if degraded

    flags = []
    if degraded:
        flags.append("degraded_input")
    if degraded and dm["r2"] > thr["r2"]:
        flags.append("detail_on_blocks")
    if link < thr["link"]:
        flags.append("weak_colour_linkage")
    if patchy > thr["patchy"]:
        flags.append("patchy_noise_possible_composite")
    if exif and "MoT-Depixelator" in exif.get("software", ""):
        flags.append("synthetic_restoration")

    if "detail_on_blocks" in flags or "patchy_noise_possible_composite" in flags or "synthetic_restoration" in flags:
        label = "manipulated_suspected"
    elif "degraded_input" in flags or "weak_colour_linkage" in flags:
        label = "inconclusive"
    else:
        label = "no_layer7_objection"                   # NOT proof of authenticity

    return dict(label=label, flags=flags, block=b, grid_offset=(grid["px"], grid["py"]),
                block_strength=round(strength, 2), flat_tiles=round(flat, 3),
                palette_colors=palette_colors(rgb), colour_linkage=round(link, 3),
                noise_patchiness=round(patchy, 3),
                demux={k: (round(v, 3) if isinstance(v, float) else v) for k, v in dm.items()},
                restoration=rinfo, restored=restored)


# --------------------------------------------------------------------------
# 7.7  Fail-closed verdict (veto rule instead of averaging)
# --------------------------------------------------------------------------
HARD_LAYERS = ("exif", "chroma")      # hard-evidence layers: pixels alone cannot fix them


def final_verdict(l7, layers=None, c2pa="absent", thr=None):
    thr = {**DEFAULT_THR, **(thr or {})}
    layers = layers or {}
    flags = l7["flags"]
    reasons = []

    # trust weights: lower the layers that degraded input is known to fool
    w = dict(neural=1.0, fingerprint=1.0, exif=1.0, chroma=1.0)
    if "degraded_input" in flags:
        w["neural"] = w["fingerprint"] = 0.2
        reasons.append("degraded input: neural/fingerprint trust lowered to 0.2")

    naive = weighted = None
    given = {k: float(v) for k, v in layers.items() if k in w}
    if given:
        naive = sum(given.values()) / len(given)
        weighted = sum(w[k] * v for k, v in given.items()) / sum(w[k] for k in given)
        
        # If this is a synthetic restoration by our tool, the neural scores are invalid 
        # (the high frequencies are lost/synthetic). Force the scores to 0.0 (100% Fake/AI).
        if "synthetic_restoration" in flags:
            naive = 0.0
            weighted = 0.0
            reasons.append("synthetic restoration: trust scores overridden to 0.0 (FAKE/AI)")

    vetoed = [k for k in HARD_LAYERS if k in given and given[k] < thr["veto"]]
    if c2pa == "invalid":
        vetoed.append("c2pa")

    if l7["label"] == "manipulated_suspected":
        label = "manipulated_suspected"
        reasons.append("Layer 7 flags: " + ", ".join(flags))
    elif vetoed:
        label = "likely_synthetic_or_manipulated"
        reasons.append("veto by hard-evidence layer(s): " + ", ".join(vetoed))
    elif c2pa == "valid":
        label = "verified_authentic"
        reasons.append("valid C2PA chain and no veto")
    elif l7["label"] == "inconclusive":
        label = "inconclusive"
        reasons.append("degraded input or weak colour linkage: other layers unreliable")
    else:
        label = "no_objection"
        reasons.append("no flags, but no C2PA proof: this is NOT proof of authenticity")

    return dict(label=label, reasons=reasons,
                naive_average=None if naive is None else round(naive, 3),
                trust_weighted_score=None if weighted is None else round(weighted, 3))


# --------------------------------------------------------------------------
# Running on files, reporting
# --------------------------------------------------------------------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def collect_paths(items):
    out = []
    for it in items:
        if os.path.isdir(it):
            for f in sorted(os.listdir(it)):
                if f.lower().endswith(IMG_EXT):
                    out.append(os.path.join(it, f))
        elif os.path.isfile(it):
            out.append(it)
        else:
            print(f"[skip] not found: {it}", file=sys.stderr)
    return out


def run_file(path, args, thr):
    rgb, exif, note = load_image(path, thr["max_side"])
    full = None
    if note:                                             # analysis ran on a centre crop: restore the whole frame
        img = Image.open(path)
        if img.width * img.height <= 16_000_000:
            full = np.asarray(img.convert("RGB"))
    l7 = analyse(rgb, thr, exif, full=full)
    verdict = final_verdict(l7, args.layers, args.c2pa, thr)
    restored = l7.pop("restored")
    saved = None
    if restored is not None and args.restored_out:
        os.makedirs(args.restored_out, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        saved = os.path.join(args.restored_out, f"{stem}_restored.png")
        from PIL.PngImagePlugin import PngInfo
        metadata = PngInfo()
        metadata.add_text("Software", "MoT-Depixelator")
        Image.fromarray(restored).save(saved, pnginfo=metadata)
    return dict(file=path, note=note, exif=exif, layer7=l7, verdict=verdict, restored_copy=saved)


def print_report(r):
    l7, v = r["layer7"], r["verdict"]
    print("=" * 70)
    print(f"File     : {r['file']}" + (f"   ({r['note']})" if r["note"] else ""))
    print(f"VERDICT  : {v['label'].upper()}")
    for why in v["reasons"]:
        print(f"  - {why}")
    if v["naive_average"] is not None:
        print(f"Scores   : naive average = {v['naive_average']}   "
              f"trust-weighted = {v['trust_weighted_score']}   (the verdict ignores averages)")
    print(f"Layer 7  : {l7['label']}   flags={l7['flags']}")
    print(f"  grid block={l7['block']} strength={l7['block_strength']} offset={l7['grid_offset']} "
          f"flat_tiles={l7['flat_tiles']} palette_colors={l7['palette_colors']}")
    print(f"  demux R2={l7['demux']['r2']} cross_corr={l7['demux']['cross_corr']} "
          f"| colour_linkage={l7['colour_linkage']} | noise_patchiness={l7['noise_patchiness']}")
    e = r["exif"]
    print(f"  EXIF present={e['present']} make={e['make']} model={e['model']}")
    ri = l7.get("restoration") or {}
    if ri.get("method"):
        print(f"  restoration: {ri['method']}  block={ri.get('block')} scale={ri.get('scale')} "
              f"preset={ri.get('preset')}" + (f"  error={ri['error']}" if ri.get("error") else ""))
    if r["restored_copy"]:
        print(f"  restored copy saved -> {r['restored_copy']}  (run your layers 1-5 on it too)")


# --------------------------------------------------------------------------
# Self-test (synthetic images; proves the code runs, NOT real-world accuracy)
# --------------------------------------------------------------------------
def _pixelate(img, b):
    h, w = img.shape[:2]
    s = img[:h // b * b, :w // b * b].astype(float).reshape(h // b, b, w // b, b, 3).mean((1, 3))
    return np.kron(s, np.ones((b, b, 1))).astype(np.uint8)


def selftest():
    rng = np.random.default_rng(0)
    H = W = 256
    scene = np.stack([gaussian_filter(rng.normal(size=(H, W)), 12) for _ in range(3)], -1)
    scene = (scene - scene.min()) / (scene.max() - scene.min()) * 200 + 20
    scene[100:160, 80:200] += 30
    lum = rng.normal(0, 4, (H, W, 1))
    col = rng.normal(0, 1.2, (H, W, 3))
    camera = np.clip(scene + lum + col, 0, 255).astype(np.uint8)

    pix = _pixelate(camera, 8)
    detail = camera.astype(float) - gaussian_filter(camera.astype(float), (1.5, 1.5, 0))
    pix_detail = np.clip(pix + 0.6 * detail, 0, 255).astype(np.uint8)
    patched = camera.copy()
    patched[60:120, 60:120] = np.clip(
        gaussian_filter(camera[60:120, 60:120].astype(float), (2.5, 2.5, 0)), 0, 255)
    shifted = pix[5:, 3:]                                   # pixelated + cropped (grid offset)

    cases = [
        ("camera-like", camera, "no_layer7_objection", None),
        ("pixelated b=8", pix, "inconclusive", 8),
        ("pixelated + detail", pix_detail, "manipulated_suspected", 8),
        ("patched region", patched, "manipulated_suspected", None),
        ("pixelated, cropped offset", shifted, "inconclusive", 8),
    ]
    ok = True
    print("Layer 7 self-test (synthetic images)")
    print("-" * 70)
    for name, im, want, want_b in cases:
        r = analyse(im)
        good = r["label"] == want and (want_b is None or r["block"] == want_b)
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:26s} -> {r['label']:22s} "
              f"block={r['block']:<3d} flags={r['flags']}")

    print("-" * 70)
    print("Verdict test: the '89 % real' situation from your notes")
    layers = {"neural": 0.95, "fingerprint": 0.95, "exif": 0.10, "chroma": 0.10}
    r = analyse(pix)
    v = final_verdict(r, layers, "absent")
    good = v["label"] != "no_objection" and v["label"] != "verified_authentic"
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  naive average={v['naive_average']} "
          f"(looks fine)  ->  verdict={v['label']}")
    for why in v["reasons"]:
        print("       -", why)
    print("-" * 70)
    print("Demultiplex test: mix a synthetic photo the way pixelate2 does, then restore it")
    yy, xx = np.mgrid[0:H, 0:W]
    photo = scene + 25 * np.sin(xx / 5.0 + yy / 9.0)[..., None] + 30 * ((xx // 37 + yy // 29) % 2)[..., None]
    photo = np.clip(photo + rng.normal(0, 2, (H, W, 3)), 0, 255)
    truth = _undo_prep(photo, MUX_PRESETS[4]["clarity"])
    for name, shape in (("mosaic + detail (b=4)", (H, W)), ("same, odd size 253x251", (253, 251))):
        src = np.ascontiguousarray(np.moveaxis(photo[:shape[0], :shape[1]], 2, 0)).astype(_F32)
        m = _Mux(shape[0], shape[1], 4, MUX_PRESETS[4])
        mixed = _unsharp(np.clip(m.S(src) + m.KG(src), 0, 255), 1.2)
        mixed = np.round(np.moveaxis(mixed, 0, 2)).astype(np.uint8)
        info = {}
        out = restore(mixed, info=info)
        ref = truth[:shape[0], :shape[1]]
        e0 = np.sqrt(((mixed - ref) ** 2).mean())
        e1 = np.sqrt(((out - ref) ** 2).mean())
        good = info["method"] == "demultiplex" and info["block"] == 4 and e1 < 0.5 * e0
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:26s} -> {info['method']:12s} block={info['block']} "
              f"error vs source: {e0:.1f} -> {e1:.1f} grey levels")
    info = {}
    out = restore(pix, info=info)
    good = info["method"] == "interpolate" and info["block"] == 8
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {'plain mosaic (b=8)':26s} -> {info['method']:12s} block={info['block']}")
    print("-" * 70)
    print("ALL SELF-TESTS PASSED" if ok else "SOME SELF-TESTS FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Layer 7: degradation-aware forensic layer")
    p.add_argument("paths", nargs="*", help="image files or folders")
    p.add_argument("--layers", type=json.loads, default=None,
                   help='JSON of P(real) from your layers, e.g. \'{"neural":0.9,"exif":0.1}\'')
    p.add_argument("--c2pa", choices=["valid", "invalid", "absent"], default="absent")
    p.add_argument("--thresholds", help="JSON file overriding DEFAULT_THR")
    p.add_argument("--restored-out", help="folder to save restored (depixelated) copies")
    p.add_argument("--json", help="write full report to this JSON file")
    p.add_argument("--selftest", action="store_true", help="run built-in synthetic tests")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.paths:
        print("Give image paths or use --selftest.  See --help.", file=sys.stderr)
        sys.exit(2)
    thr = dict(DEFAULT_THR)
    if args.thresholds:
        with open(args.thresholds) as f:
            thr.update(json.load(f))
    results = []
    for path in collect_paths(args.paths):
        try:
            r = run_file(path, args, thr)
        except Exception as e:                              # keep going on bad files
            print(f"[error] {path}: {e}", file=sys.stderr)
            continue
        results.append(r)
        print_report(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nJSON report written to {args.json}")


if __name__ == "__main__":
    main()