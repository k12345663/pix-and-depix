#!/usr/bin/env python3
import sys
import json
import numpy as np
from PIL import Image
from de import to_gray, estimate_grid, flat_tile_ratio, DEFAULT_THR

def is_pixelated(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        rgb = np.asarray(img)
        
        # Handle large images by cropping the center (same as de.py)
        h, w = rgb.shape[:2]
        max_side = DEFAULT_THR["max_side"]
        if max(h, w) > max_side:
            ch, cw = min(h, max_side), min(w, max_side)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            rgb = rgb[y0:y0 + ch, x0:x0 + cw]
            
        gray = to_gray(rgb.astype(float))
        
        grid = estimate_grid(gray)
        b, strength = grid["b"], grid["strength"]
        
        gray_c = gray[grid["py"]:, grid["px"]:]
        flat = flat_tile_ratio(gray_c, b)
        
        # Using relaxed thresholds for screenshots
        is_degraded = strength > DEFAULT_THR["block"] or (b >= 2 and strength > 1.15 and flat > DEFAULT_THR["flat"]) or (flat > 0.20)
        
        return {
            "pixelated": bool(is_degraded),
            "block_size": int(b) if is_degraded else None,
            "strength": round(float(strength), 3),
            "flat_ratio": round(float(flat), 3)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pixelated.py <image_path>")
        sys.exit(1)
        
    result = is_pixelated(sys.argv[1])
    print(json.dumps(result, indent=2))
