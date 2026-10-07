# Orthogonal Multiplexing for Image Pixelation: Project Report

This document outlines the architecture, mathematics, and forensic evasion techniques developed for the `pixelate2.py` pipeline. The goal of this project was to create a pixelation algorithm that heavily obscures the image mathematically (protecting privacy) while remaining highly recognizable to human perception, all while bypassing advanced AI forensic manipulation detectors.

---

## 1. The Core Scripts

### A. `pixelate2.py` (The Multiplexer)
This is the heart of the project. Standard pixelation destroys mid-to-high frequency data (edges, shapes, text). To make the image recognizable without reverting the pixel-art aesthetic, this script mathematically dissects the image into four orthogonal domains, extracts specific details, and multiplexes them back together.

### B. `pure_pix.py` (The Control)
A standard, brute-force 4-pixel block downsample-upsample script. We built this simply to act as a scientific control to test against the forensic detector.

### C. `de.py` (The Verifier)
A Layer 7 forensic image verification script designed to catch "synthetic or manipulated" images. It hunts for detail hidden inside pixel blocks (demultiplexing), unnatural color channel linkages, and patchy noise (composite splicing).

---

## 2. The Mathematics of `pixelate2.py`

To inject detail back into a blocky image without destroying the blocks, we must use **Orthogonal Domain Multiplexing**. Because the spatial, frequency, and transform domains are mathematically orthogonal (they measure entirely different properties of light and structure), signals embedded in them do not destructively interfere.

### Branch 1: Spatial Domain (The Base)
*   **Math:** Nearest-neighbor interpolation and Adaptive Palette Quantization (K-Means).
*   **Action:** Reduces the image into discrete $N \times N$ block step-functions and compresses the color space.

### Branch 2: Fourier Domain (High-Pass Edges)
*   **Math:** 2D Fast Fourier Transform ($FFT_{2D}$) and Gaussian Convolution.
*   **Action:** The image is converted into a frequency spectrum. A Gaussian curve centered at the DC (zero-frequency) component zeroes out high frequencies. The Inverse FFT yields a perfectly blurry image. Subtracting this from the original image ($Original - LowPass$) perfectly isolates the high-frequency edges.

### Branch 3: Discrete Cosine Transform (Mid-Frequency Structure)
*   **Math:** Type-II 2D DCT and Hard Thresholding.
*   **Action:** DCT breaks the image into oscillating cosine waves. We create a mask that zeros out 85% of the matrix, keeping only the lowest 15% (the "structural skeleton"). The difference ($Original - InverseDCT(Skeleton)$) isolates mid-frequency details (facial contours, soft gradients) that pixelation destroys but Fourier edge-detection misses.

### Branch 4: Haar Wavelet Domain (Directional Gradients)
*   **Math:** 2D Discrete Wavelet Transform (DWT).
*   **Action:** The Haar transform calculates sums and differences of adjacent pixels, decomposing the image into 4 sub-bands: LL, LH, HL, and HH. We discard the base (LL) and boost the LH (horizontal), HL (vertical), and HH (diagonal) bands. Inverse DWT injects directionally-aware edge structures.

### Multiplexed Reconstruction
The final image is a linear combination of these domains:
`Output = SpatialBase + (α * Fourier) + (β * DCT) + (γ * Wavelet)`

---

## 3. Defeating the AI Detector (`de.py`)

When running our highly structured image through the Layer 7 detector, we encountered three massive forensic red flags. Here is how we mathematically solved them:

### Problem 1: "Bayer CFA Pattern Confirmed" / "Mosaic Jitter"
*   **The Cause:** Our Fourier and Wavelet transforms were *too* precise. They extracted the microscopic raw sensor noise and the Color Filter Array (CFA) demosaicing pattern left by the original camera, boosted it, and injected it onto synthetic pixel art. The detector saw raw camera noise on a synthetic image and flagged it.
*   **The Mathematical Fix:** Counter-forensics. We applied a tiny Gaussian blur ($\sigma=0.75$) to the original image *before* feature extraction. This acts as a low-pass filter that obliterates the PRNU (Photo Response Non-Uniformity) sensor noise floor, ensuring we only extracted true physical edges.

### Problem 2: "Chrominance Edge Disconnect"
*   **The Cause:** Initially, we extracted structural detail only from the Luminance (grayscale) channel to avoid "rainbow artifacts." However, this meant the brightness had sharp edges while the RGB color channels remained perfectly flat blocks. Detectors flagged this luma/chroma misalignment as "color without structure."
*   **The Mathematical Fix:** Because our CFA pre-blur destroyed the sensor noise that was causing the rainbow artifacts, we safely reverted the pipeline to extract and inject Fourier/DCT/Wavelet details across all 3 RGB channels independently. The colors now perfectly track the structural edges.

### Problem 3: "Unusual Patterns" (Mathematical Ringing)
*   **The Cause:** The final stage of our script applied an `UnsharpMask` with a threshold of `1`. Frequency transforms naturally leave microscopic mathematical ripples (ringing). A threshold of `1` forced the mask to aggressively amplify these invisible ripples into harsh, artificial patterns that screamed "algorithmically generated."
*   **The Mathematical Fix:** We raised the UnsharpMask threshold to `12`. This forced the algorithm to ignore the subtle mathematical noise and *only* sharpen major physical contrast boundaries (like pixel blocks and facial features). 

---

## 4. The Final "Patchy Noise" Anomaly

After fixing all mathematical artifacts, `de.py` still flagged our image (and the `pure_pix.py` control image!) as `patchy_noise_possible_composite`.

### The Diagnosis
The `de.py` detector calculates `noise_patchiness` by measuring the standard deviation of high-frequency energy across 32x32 tiles. 
1. If the image has a sharp subject and a flat/blurry background, the high-frequency energy is massive in some tiles and zero in others.
2. The detector's placeholder threshold for this is `0.12` (which demands an almost perfectly uniform field of static noise everywhere).
3. Furthermore, because our block size was `4`, `de.py` failed to classify the grid as "degraded" (requiring a strength of > 1.5, whereas standard pixelation only generated 1.41).

### The Conclusion
Our math successfully bypassed the `demultiplex` check (`demux R2 = 0.0`). The `patchy_noise` flag is a false positive caused by the miscalibrated placeholder thresholds in the `de.py` script, which unfairly flags natural subject-background separation as a composite splice. The mathematical pixelation pipeline is completely sound.
