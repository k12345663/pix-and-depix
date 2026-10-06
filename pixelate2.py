"""
pixelate2.py – Recognisability-preserving multiplexed pixel-art generator.

Pipeline (orthogonal domain multiplexing):
  ┌─────────────────────────────────────────────────────────────────┐
  │  ORIGINAL IMAGE                                                │
  │     ├── Spatial branch   → pixelate + palette reduction        │
  │     ├── Fourier branch   → high-pass edge map (detail)         │
  │     ├── DCT branch       → mid-freq structure map              │
  │     └── Wavelet branch   → Haar multi-scale edge detail        │
  │                                                                │
  │  RECONSTRUCTION = spatial_base                                 │
  │                 + α · fourier_edges                             │
  │                 + β · dct_structure                             │
  │                 + γ · wavelet_detail                            │
  │                 → re-quantize → sharpen → output               │
  └─────────────────────────────────────────────────────────────────┘

Each branch operates in a mathematically orthogonal domain so the
injected detail does not destructively interfere with the base.

Install: pip install pillow numpy scipy
"""
import argparse, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, dctn, idctn
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# ─────────────────────────── constants ───────────────────────────
PRESETS = {
    #                                          edge  struct  wavelet
    "clear":  dict(size=4,  colors=128, clarity=2.0, outline=False,
                   alpha=0.35, beta=0.20, gamma=0.15),
    "retro":  dict(size=8,  colors=48,  clarity=1.5, outline=True,
                   alpha=0.40, beta=0.25, gamma=0.20),
    "poster": dict(size=6,  colors=32,  clarity=2.5, outline=False,
                   alpha=0.30, beta=0.15, gamma=0.10),
}

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


# ─────────────────────────── helpers ─────────────────────────────
def has_transparency(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )


def prep(img: Image.Image, clarity: float) -> Image.Image:
    """Pre-process: autocontrast + contrast/colour boost."""
    if clarity <= 0:
        return img
    img = ImageOps.autocontrast(img, cutoff=0.3, preserve_tone=True)
    img = ImageEnhance.Contrast(img).enhance(1 + 0.08 * clarity)
    return ImageEnhance.Color(img).enhance(1 + 0.15 * clarity)


def _norm8(arr: np.ndarray) -> np.ndarray:
    """Normalise a float array to uint8 [0, 255]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return (255.0 * (arr - lo) / (hi - lo)).astype(np.uint8)


# ─────────────── BRANCH 1: Spatial (pixelation) ─────────────────
def pixelate(img: Image.Image, size: int) -> Image.Image:
    """Down-sample → up-sample with nearest-neighbour for hard pixel edges."""
    w, h = img.size
    small = img.resize((max(1, w // size), max(1, h // size)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def reduce_palette(img: Image.Image, colors: int) -> Image.Image:
    """Quantize to a limited adaptive palette."""
    return img.convert("P", palette=Image.ADAPTIVE, colors=colors, dither=Image.NONE).convert("RGB")


def add_outline(img: Image.Image) -> Image.Image:
    """Overlay dark edge lines for a retro look."""
    edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
    edges = ImageOps.invert(edges).point(lambda p: 255 if p > 30 else 0, mode="1")
    return Image.composite(img, Image.new("RGB", img.size, "black"), edges)


def spatial_branch(img: Image.Image, size: int, colors: int,
                   outline: bool) -> np.ndarray:
    """Full spatial-domain chain → float64 array [0, 255]."""
    out = pixelate(img, size)
    out = reduce_palette(out, colors)
    if outline:
        out = add_outline(out)
    return np.asarray(out).astype(np.float64)


# ──────────── BRANCH 2: Fourier high-pass edge map ──────────────
def _gaussian_kernel_2d(rows: int, cols: int, sigma: float) -> np.ndarray:
    """2-D Gaussian centred at DC (for low-pass in frequency domain)."""
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[-cy:rows - cy, -cx:cols - cx]
    return np.exp(-(X * X + Y * Y) / (2.0 * sigma * sigma))


def fourier_highpass(channel: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Extract edges via Fourier: original − Gaussian-low-pass = high-pass.

    Returns a signed float array (positive = bright edge, negative = dark edge).
    """
    F = fftshift(fft2(channel))
    lpf = _gaussian_kernel_2d(*channel.shape, sigma)
    low = np.real(ifft2(ifftshift(F * lpf)))
    return channel - low  # high-pass residual


def fourier_edge_map(img_arr: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Per-channel Fourier high-pass → combined edge detail (same shape as input)."""
    out = np.zeros_like(img_arr)
    for ch in range(img_arr.shape[2]):
        out[:, :, ch] = fourier_highpass(img_arr[:, :, ch], sigma)
    return out


# ──────────── BRANCH 3: DCT mid-frequency structure ─────────────
def dct_structure(channel: np.ndarray, keep_ratio: float = 0.15) -> np.ndarray:
    """Keep only the lowest `keep_ratio` fraction of DCT coefficients.

    This captures the mid-frequency structural skeleton of the image:
    global shapes, large gradients, face outlines, etc.
    """
    D = dctn(channel, norm="ortho")
    rows, cols = D.shape
    # Zero out everything beyond the keep window
    r_keep = max(1, int(rows * keep_ratio))
    c_keep = max(1, int(cols * keep_ratio))
    mask = np.zeros_like(D)
    mask[:r_keep, :c_keep] = 1.0
    return idctn(D * mask, norm="ortho")


def dct_structure_map(img_arr: np.ndarray, keep_ratio: float = 0.15) -> np.ndarray:
    """Per-channel DCT structure extraction → difference from original."""
    skeleton = np.zeros_like(img_arr)
    for ch in range(img_arr.shape[2]):
        skeleton[:, :, ch] = dct_structure(img_arr[:, :, ch], keep_ratio)
    return img_arr - skeleton  # mid-freq detail


# ──────────── BRANCH 4: Haar wavelet edge detail ────────────────
def _haar_decompose(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
    """One-level 2-D Haar wavelet decomposition.

    Returns (LL, LH, HL, HH) each at half resolution.
    """
    # Row-wise transform
    rows, cols = channel.shape
    r = rows - rows % 2  # ensure even
    c = cols - cols % 2
    ch = channel[:r, :c]

    lo = (ch[:, 0::2] + ch[:, 1::2]) / 2.0
    hi = (ch[:, 0::2] - ch[:, 1::2]) / 2.0

    # Column-wise on lo
    LL = (lo[0::2, :] + lo[1::2, :]) / 2.0
    LH = (lo[0::2, :] - lo[1::2, :]) / 2.0
    # Column-wise on hi
    HL = (hi[0::2, :] + hi[1::2, :]) / 2.0
    HH = (hi[0::2, :] - hi[1::2, :]) / 2.0
    return LL, LH, HL, HH


def _haar_reconstruct(LL, LH, HL, HH) -> np.ndarray:
    """Inverse one-level 2-D Haar wavelet."""
    r2, c2 = LL.shape
    rows, cols = r2 * 2, c2 * 2

    lo = np.zeros((rows, c2))
    lo[0::2, :] = LL + LH
    lo[1::2, :] = LL - LH

    hi = np.zeros((rows, c2))
    hi[0::2, :] = HL + HH
    hi[1::2, :] = HL - HH

    out = np.zeros((rows, cols))
    out[:, 0::2] = lo + hi
    out[:, 1::2] = lo - hi
    return out


def wavelet_detail_map(img_arr: np.ndarray, boost: float = 2.0) -> np.ndarray:
    """Extract edge detail via Haar wavelet, boost it, reconstruct per channel."""
    out = np.zeros_like(img_arr)
    for ch in range(img_arr.shape[2]):
        channel = img_arr[:, :, ch]
        LL, LH, HL, HH = _haar_decompose(channel)
        recon = _haar_reconstruct(np.zeros_like(LL), LH * boost, HL * boost, HH * boost)
        out[:recon.shape[0], :recon.shape[1], ch] = recon
    return out


# ──────────────── MULTIPLEXED RECONSTRUCTION ─────────────────────
def multiplex_reconstruct(
    img: Image.Image,
    size: int,
    colors: int,
    clarity: float,
    outline: bool,
    alpha: float = 0.35,   # Fourier edge injection strength
    beta: float  = 0.20,   # DCT structure injection strength
    gamma: float = 0.15,   # Wavelet detail injection strength
) -> Image.Image:
    """Reconstruct a recognisable pixel-art image via orthogonal-domain
    multiplexing.

    Mathematical domains used (all orthogonal to each other):
    ─────────────────────────────────────────────────────────
    1. Spatial        — nearest-neighbour pixelation + palette
    2. Fourier        — high-pass edge residual (original − Gaussian LP)
    3. DCT            — mid-frequency structural detail
    4. Haar wavelet   — directional edge detail (H / V / diagonal)

    Reconstruction formula:
        output = spatial_base + α·fourier_edges + β·dct_detail + γ·wavelet_edges
    """
    original = np.asarray(img).astype(np.float64)
    
    # Denoise the original to kill Bayer CFA patterns and high-freq sensor noise
    denoised_img = img.filter(ImageFilter.GaussianBlur(radius=0.75))
    original_denoised = np.asarray(denoised_img).astype(np.float64)

    # ── Branch 1: Spatial base ──
    base = spatial_branch(img, size, colors, outline)

    # ── Branch 2: Fourier high-pass edges ──
    f_edges = fourier_edge_map(original_denoised, sigma=15.0)

    # ── Branch 3: DCT mid-frequency structure ──
    d_struct = dct_structure_map(original_denoised, keep_ratio=0.15)

    # ── Branch 4: Haar wavelet directional edges ──
    w_detail = wavelet_detail_map(original_denoised, boost=2.0)

    # ── Multiplexed reconstruction ──
    # Additive injection: detail is ADDED to the pixelated base
    result = (
        base
        + alpha * f_edges
        + beta  * d_struct
        + gamma * w_detail
    )

    # Clip and convert back
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    # Final crispness boost (unsharp mask)
    if clarity > 0:
        result_img = result_img.filter(
            ImageFilter.UnsharpMask(
                radius=1.2, percent=int(60 * clarity), threshold=12
            )
        )

    # Light re-quantize to keep pixel-art feel but with more colour headroom
    final_colors = min(256, int(colors * 1.5))
    result_img = reduce_palette(result_img, final_colors)

    return result_img


# ─────────────────────── I/O + CLI ───────────────────────────────
def process_image(path: Path, preset: dict) -> None:
    """Load → transform → save."""
    try:
        img = Image.open(path)
        exif = img.getexif()
    except Exception as e:
        sys.stderr.write(f'Failed to open "{path}": {e}\n')
        return

    alpha = None
    if has_transparency(img):
        img = img.convert("RGBA")
        alpha = img.split()[-1]
    
    img = img.convert("RGB")

    img = prep(img, preset["clarity"])

    result = multiplex_reconstruct(
        img,
        size=preset["size"],
        colors=preset["colors"],
        clarity=preset["clarity"],
        outline=preset["outline"],
        alpha=preset.get("alpha", 0.35),
        beta=preset.get("beta", 0.20),
        gamma=preset.get("gamma", 0.15),
    )

    if alpha:
        # Resize alpha to match result size if needed (though it shouldn't change)
        if alpha.size != result.size:
            alpha = alpha.resize(result.size, Image.NEAREST)
        result.putalpha(alpha)

    out_path = path.with_name(path.stem + "_pixelated" + path.suffix)
    
    save_kwargs = {}
    if exif:
        save_kwargs["exif"] = exif
        
    result.save(out_path, **save_kwargs)
    print(f"Saved pixelated image to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pixelate2 – recognisability-preserving multiplexed pixel-art generator."
    )
    parser.add_argument("image", type=Path, help="Path to the source image.")
    parser.add_argument(
        "-p", "--preset",
        choices=PRESETS.keys(),
        default="clear",
        help="Choose a preset (default: clear).",
    )
    args = parser.parse_args()
    process_image(args.image, PRESETS[args.preset])


if __name__ == "__main__":
    main()
