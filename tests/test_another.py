"""
Simple Test Script for Image Reconstruction Module
--------------------------------------------------
Runs reconstruction using both 'simple' and 'blended' methods
on one pair of test images and masks.

Usage:
    python tests/test_reconstruction_simple.py
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from modules.reconstruction_module import ReconstructionModule


# =======================
# CONFIGURATION
# =======================
DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "test_inputs"
OUTPUT_DIR = DATA_DIR / "test_results"


# =======================
# HELPER FUNCTIONS
# =======================

def load_images(img1_path, img2_path, mask1_path, mask2_path):
    """Load test images and masks (any common extension)."""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    mask1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(str(mask2_path), cv2.IMREAD_GRAYSCALE)

    if any(x is None for x in [img1, img2, mask1, mask2]):
        raise FileNotFoundError("❌ Could not load one or more input files.")

    print(f"✓ Loaded images: {img1.shape}, {img2.shape}")
    return img1, img2, mask1, mask2


def save_results(result_dir, img1, img2, mask1, mask2, result, overlap_mask, visualization, stats, method):
    """Save all results (image outputs + stats)."""
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    cv2.imwrite(str(result_dir / "input_img1.png"), img1)
    cv2.imwrite(str(result_dir / "input_img2.png"), img2)
    cv2.imwrite(str(result_dir / "input_mask1.png"), mask1)
    cv2.imwrite(str(result_dir / "input_mask2.png"), mask2)
    cv2.imwrite(str(result_dir / "result.png"), result)
    cv2.imwrite(str(result_dir / "overlap_mask.png"), overlap_mask)
    cv2.imwrite(str(result_dir / "visualization.png"), visualization)

    # Save statistics
    stats_path = result_dir / "stats.txt"
    with open(stats_path, "w") as f:
        f.write("Image Reconstruction Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: {method}\n\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print(f"💾 Results saved in {result_dir}\n")


def get_next_test_number():
    """Automatically increment test number without overwriting previous results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT_DIR.glob("testresult_*"))
    nums = []
    for folder in existing:
        try:
            nums.append(int(folder.name.split("_")[-1]))
        except:
            pass
    return max(nums, default=0) + 1


# =======================
# MAIN TEST FUNCTION
# =======================

def main():
    print("\n" + "="*60)
    print("IMAGE RECONSTRUCTION - SIMPLE TEST")
    print("="*60)

    # Find any valid image pair inside test_inputs
    img1_path = next(INPUT_DIR.glob("img1_*.*"), None)
    img2_path = next(INPUT_DIR.glob("img2_*.*"), None)
    mask1_path = next(INPUT_DIR.glob("mask1_*.*"), None)
    mask2_path = next(INPUT_DIR.glob("mask2_*.*"), None)

    if not all([img1_path, img2_path, mask1_path, mask2_path]):
        print("❌ Missing input files in data/test_inputs/")
        print("Expected files: img1_xx, img2_xx, mask1_xx, mask2_xx (any extension)")
        return

    img1, img2, mask1, mask2 = load_images(img1_path, img2_path, mask1_path, mask2_path)

    # Run both methods
    for method in ["simple", "blended"]:
        print(f"\n🔧 Running reconstruction using '{method}' method...")
        reconstructor = ReconstructionModule(blend_method=method)
        result, overlap_mask, stats = reconstructor.reconstruct(img1, img2, mask1, mask2)
        visualization = reconstructor.visualize_reconstruction(img1, img2, result, mask1, mask2, overlap_mask)

        test_num = get_next_test_number()
        result_dir = OUTPUT_DIR / f"testresult_{test_num}"
        save_results(result_dir, img1, img2, mask1, mask2, result, overlap_mask, visualization, stats, method)

        print(f"📊 Stats for '{method}':")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("-" * 60)

    print("\n✅ All tests complete! Check 'data/test_results/' for outputs.")


if __name__ == "__main__":
    main()
