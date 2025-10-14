"""
Enhanced Glare Detection Module for Document Image Processing
Complete implementation with edge handling improvements
"""

import cv2
import numpy as np
import os
import json
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class EnhancedGlareDetector:
    """
    Advanced glare detector with superior edge pixel handling
    Includes 6 major improvements over basic detection
    """

    def __init__(self, config_file='glare_config.json'):
        """
        Initialize detector with settings from config file if available

        Args:
            config_file: Path to JSON config file for saving/loading settings
        """
        self.config_file = config_file
        self.min_area = 50

        # Load settings from file or use defaults
        self.load_settings()

        # Morphological kernels for cleaning up the mask
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))

    def load_settings(self):
        """Load settings from config file if it exists, otherwise use defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.hsv_lower = np.array(config['hsv_lower'])
                    self.hsv_upper = np.array(config['hsv_upper'])
                    self.min_area = config.get('min_area', 50)
                    # Load edge handling params if saved
                    self.edge_expansion_pixels = config.get(
                        'edge_expansion', 10)
                    self.gradient_threshold = config.get(
                        'gradient_threshold', 30)
                    self.feather_radius = config.get('feather_radius', 15)
                    print(f"\n✅ Loaded saved settings from {self.config_file}")
                    print(f"   V_min (brightness): {self.hsv_lower[2]}")
                    print(f"   S_max (saturation): {self.hsv_upper[1]}")
                    print(f"   Min Area: {self.min_area} pixels")
                    print(
                        f"   Edge Expansion: {self.edge_expansion_pixels} pixels")
            except Exception as e:
                print(f"\n⚠️  Error loading config: {e}")
                print(f"   Using default settings")
                self._set_default_settings()
        else:
            print(f"\nℹ️  No saved settings found. Using defaults.")
            self._set_default_settings()

    def _set_default_settings(self):
        """Set default detection thresholds"""
        self.hsv_lower = np.array([0, 0, 200])    # [H_min, S_min, V_min]
        self.hsv_upper = np.array([180, 30, 255])  # [H_max, S_max, V_max]
        self.min_area = 50
        self.edge_expansion_pixels = 10
        self.gradient_threshold = 30
        self.feather_radius = 15

    def save_settings(self):
        """Save current settings to config file"""
        config = {
            'hsv_lower': self.hsv_lower.tolist(),
            'hsv_upper': self.hsv_upper.tolist(),
            'min_area': int(self.min_area),
            'edge_expansion': int(self.edge_expansion_pixels),
            'gradient_threshold': int(self.gradient_threshold),
            'feather_radius': int(self.feather_radius)
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"\n💾 Settings permanently saved to {self.config_file}")
            print(f"   V_min (brightness): {self.hsv_lower[2]}")
            print(f"   S_max (saturation): {self.hsv_upper[1]}")
            print(f"   Min Area: {self.min_area} pixels")
            print(f"   Edge Expansion: {self.edge_expansion_pixels} pixels")
            print(f"\n✨ These settings will be used automatically next time!")
        except Exception as e:
            print(f"\n❌ Error saving config: {e}")

    # ========== BASIC DETECTION METHODS ==========

    def detect_glare_basic(self, image: np.ndarray, method='hsv') -> np.ndarray:
        """
        Basic glare detection without edge improvements

        Args:
            image: Input BGR image
            method: 'hsv', 'brightness', or 'adaptive'

        Returns:
            Binary mask: 255 = glare, 0 = no glare
        """
        if method == 'hsv':
            mask = self._detect_hsv(image)
        elif method == 'brightness':
            mask = self._detect_brightness(image)
        elif method == 'adaptive':
            mask = self._detect_adaptive(image)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Clean up the mask
        mask = self._refine_mask_basic(mask)

        return mask

    def _detect_hsv(self, image: np.ndarray) -> np.ndarray:
        """HSV-based detection: find bright, low-saturation pixels"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask

    def _detect_brightness(self, image: np.ndarray) -> np.ndarray:
        """Brightness-based detection: find top 15% brightest pixels"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Otsu's method
        _, mask_otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Top 15% brightness
        threshold_value = np.percentile(gray, 85)
        _, mask_percentile = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Combine both
        mask = cv2.bitwise_and(mask_otsu, mask_percentile)
        return mask

    def _detect_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Combines multiple detection methods"""
        # Method 1: HSV
        mask_hsv = self._detect_hsv(image)

        # Method 2: Brightness
        mask_bright = self._detect_brightness(image)

        # Method 3: Saturation check
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask_sat = ((s < self.hsv_upper[1]) & (
            v > self.hsv_lower[2])).astype(np.uint8) * 255

        # Combine all three methods
        mask_combined = cv2.bitwise_or(mask_hsv, mask_bright)
        mask_combined = cv2.bitwise_or(mask_combined, mask_sat)

        return mask_combined

    def _refine_mask_basic(self, mask: np.ndarray) -> np.ndarray:
        """Basic mask refinement using morphological operations"""
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)

        # Fill holes in glare regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # Remove tiny regions
        mask = self._remove_small_regions(mask, min_area=self.min_area)

        return mask

    # ========== EDGE HANDLING IMPROVEMENTS ==========

    def detect_glare_edges(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        IMPROVEMENT 1: Gradient-based edge detection
        Detects transition zones around glare using intensity gradients
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradient magnitude using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Find edges of current mask
        edge_mask = cv2.Canny(mask, 50, 150)
        edge_mask = cv2.dilate(edge_mask, np.ones(
            (5, 5), np.uint8), iterations=1)

        # Find high gradient areas near glare edges
        high_gradient = (gradient_mag > self.gradient_threshold).astype(
            np.uint8) * 255
        edge_expansion = cv2.bitwise_and(high_gradient, edge_mask)

        return edge_expansion

    def expand_mask_adaptive(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        IMPROVEMENT 2: Adaptive edge expansion based on local brightness
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute local brightness
        local_brightness = cv2.GaussianBlur(gray, (15, 15), 0)
        brightness_normalized = local_brightness / 255.0

        # Create distance transform from mask edges
        dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)

        # Adaptive expansion
        expansion_threshold = self.edge_expansion_pixels * brightness_normalized
        expansion_map = (dist_transform < expansion_threshold).astype(
            np.uint8) * 255

        # Combine with original mask
        expanded_mask = cv2.bitwise_or(mask, expansion_map)

        return expanded_mask

    def create_soft_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        IMPROVEMENT 3: Feathering/blurring edges for smooth transitions
        """
        # Create distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Normalize to 0-255 range with feathering
        dist_normalized = np.clip(
            dist / self.feather_radius * 255, 0, 255).astype(np.uint8)

        # Apply Gaussian blur for smooth transition
        kernel_size = self.feather_radius * 2 + 1
        soft_mask = cv2.GaussianBlur(
            dist_normalized, (kernel_size, kernel_size), 0)

        return soft_mask

    def refine_edges_morphological(self, mask: np.ndarray) -> np.ndarray:
        """
        IMPROVEMENT 4: Advanced morphological operations for edge refinement
        """
        # Morphological gradient to find edges
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Fill the gradient back into the mask
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        # Smooth with opening
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        return refined

    def detect_bloom_effect(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        IMPROVEMENT 5: Detect bloom/halo effect around glare
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Dilate mask to get surrounding region
        dilated = cv2.dilate(mask, np.ones((21, 21), np.uint8), iterations=1)

        # Get ring around glare
        ring = cv2.bitwise_xor(dilated, mask)

        # Check brightness in ring area
        ring_pixels = gray[ring > 0]
        if len(ring_pixels) > 0:
            local_mean = np.mean(ring_pixels)
            bloom_threshold = max(local_mean + 20, 180)

            # Find bloom pixels
            bloom_mask = ((gray > bloom_threshold) & (
                ring > 0)).astype(np.uint8) * 255

            return bloom_mask

        return np.zeros_like(mask)

    # ========== INTEGRATED ENHANCED DETECTION ==========

    def detect_glare_enhanced(self, image: np.ndarray,
                              use_gradient=True,
                              use_adaptive_expansion=True,
                              use_bloom_detection=True,
                              use_edge_refinement=True,
                              return_soft=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced glare detection with all edge improvements

        Returns:
            binary_mask: Hard-edge binary mask (0 or 255)
            soft_mask: Soft-edge mask with gradual transitions (0-255) if return_soft=True
        """
        # Step 1: Basic detection
        mask = self.detect_glare_basic(image, method='hsv')

        # Step 2: Edge improvements
        if use_gradient:
            gradient_edges = self.detect_glare_edges(image, mask)
            mask = cv2.bitwise_or(mask, gradient_edges)

        if use_adaptive_expansion:
            mask = self.expand_mask_adaptive(image, mask)

        if use_bloom_detection:
            bloom = self.detect_bloom_effect(image, mask)
            mask = cv2.bitwise_or(mask, bloom)

        if use_edge_refinement:
            mask = self.refine_edges_morphological(mask)

        # Final cleanup
        mask = self._remove_small_regions(mask, self.min_area)

        # Step 3: Create soft-edge version
        if return_soft:
            soft_mask = self.create_soft_edges(mask)
            return mask, soft_mask
        else:
            return mask, mask

    def _remove_small_regions(self, mask: np.ndarray, min_area: int = 50) -> np.ndarray:
        """Remove connected components smaller than min_area"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        output = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output[labels == i] = 255

        return output

    # ========== STATISTICS & VISUALIZATION ==========

    def get_statistics(self, mask: np.ndarray) -> Dict:
        """Get statistics about detected glare"""
        glare_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage_percent = (glare_pixels / total_pixels) * 100

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        num_regions = num_labels - 1

        region_areas = [stats[i, cv2.CC_STAT_AREA]
                        for i in range(1, num_labels)]

        return {
            'glare_pixels': int(glare_pixels),
            'coverage_percent': float(coverage_percent),
            'num_regions': int(num_regions),
            'largest_region': int(max(region_areas)) if region_areas else 0
        }

    def visualize(self, image: np.ndarray, mask: np.ndarray,
                  soft_mask: np.ndarray = None, save_path: str = None):
        """Create visualization of detection results - 2x2 layout"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Detected mask (binary black and white)
        axes[0, 1].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('Glare Mask (White=Glare)',
                             fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Overlay with cyan highlighting
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 255]  # Cyan for glare
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        axes[1, 0].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Glare Highlighted (Cyan)',
                             fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Statistics with current settings
        stats = self.get_statistics(mask)
        stats_text = f"Coverage: {stats['coverage_percent']:.2f}%\n"
        stats_text += f"Regions: {stats['num_regions']}\n"
        stats_text += f"Largest: {stats['largest_region']} px\n\n"
        stats_text += f"Settings Used:\n"
        stats_text += f"V_min: {self.hsv_lower[2]}\n"
        stats_text += f"S_max: {self.hsv_upper[1]}\n"
        stats_text += f"Min Area: {self.min_area}"

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=14,
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics & Settings',
                             fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")

        plt.show()

    def visualize_comparison(self, image: np.ndarray,
                             basic_mask: np.ndarray,
                             enhanced_mask: np.ndarray,
                             soft_mask: np.ndarray = None,
                             save_path: str = None):
        """Compare basic vs enhanced edge detection - Simple 3-panel view"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Basic mask - pure binary black and white
        axes[1].imshow(basic_mask, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Basic Detection', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Enhanced mask - pure binary black and white
        axes[2].imshow(enhanced_mask, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title('Enhanced Detection', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved comparison to '{save_path}'")

        plt.show()

    # ========== INTERACTIVE TUNING ==========

    def tune_thresholds(self, image: np.ndarray):
        """
          Interactive tool to adjust detection thresholds.
          Settings are automatically saved when you press 'q' or 'Q'

          FIXED ISSUES:
          - Reduced lag with frame skipping
          - Better key detection (q, Q, ESC, s, S)
          - Window automatically gets focus
          - Instructions displayed in window
        """
        window_name = 'Glare Detection Tuning - Press Q to quit and save, ESC to quit without saving'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 700)

        # Resize image if too large (for performance)
        h, w = image.shape[:2]
        if h > 800 or w > 1200:
            scale = min(800/h, 1200/w)
            img_display = cv2.resize(image, (int(w*scale), int(h*scale)))
            print(
                f"   Image resized to {img_display.shape[:2]} for smoother tuning")
        else:
            img_display = image.copy()

        def nothing(x):
            pass

        # Create trackbars with CURRENT settings
        cv2.createTrackbar('V_min', window_name, int(
            self.hsv_lower[2]), 255, nothing)
        cv2.createTrackbar('S_max', window_name, int(
            self.hsv_upper[1]), 255, nothing)
        cv2.createTrackbar('Min Area', window_name,
                           int(self.min_area), 500, nothing)
        cv2.createTrackbar('Edge Expand', window_name, int(
            self.edge_expansion_pixels), 50, nothing)
        cv2.createTrackbar('Gradient Th', window_name, int(
            self.gradient_threshold), 100, nothing)

        print("\n" + "="*70)
        print(" ENHANCED GLARE DETECTION TUNING TOOL ")
        print("="*70)
        print("\n📊 Parameter Guide:")
        print("  • V_min (Brightness):  Higher = more strict (only very bright glare)")
        print("  • S_max (Saturation):  Lower = more strict (only white/gray glare)")
        print("  • Min Area:            Ignore glare spots smaller than this")
        print("  • Edge Expand:         How far to extend mask around detected glare")
        print("  • Gradient Th:         Sensitivity for detecting glare edges")
        print("\n⌨️  Controls:")
        print("  • Q or q      = Quit and SAVE settings permanently")
        print("  • ESC         = Quit WITHOUT saving")
        print("  • S or s      = Save current mask as PNG")
        print("  • H or h      = Show help")
        print("\n💡 Tips:")
        print("  • Start with V_min around 200-220 for documents")
        print("  • Increase Edge Expand to capture fading glare edges (15-25)")
        print("  • Lower Gradient Th to be more sensitive to edges (15-25)")
        print("  • Click on the image window to ensure it has focus")
        print("="*70)
        print("\n🎯 Tuning started... Click the window and adjust sliders\n")

        # Frame skipping for performance (update every N frames)
        frame_skip = 2
        frame_count = 0
        last_params = None
        saved_count = 0

        # Make sure window gets focus
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(100)  # Give window time to appear
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)

        while True:
            # Get current parameter values
            v_min = cv2.getTrackbarPos('V_min', window_name)
            s_max = cv2.getTrackbarPos('S_max', window_name)
            min_area = cv2.getTrackbarPos('Min Area', window_name)
            edge_expand = cv2.getTrackbarPos('Edge Expand', window_name)
            gradient_th = cv2.getTrackbarPos('Gradient Th', window_name)

            current_params = (v_min, s_max, min_area, edge_expand, gradient_th)

            # Only recompute if parameters changed (reduces lag)
            if current_params != last_params:
                frame_count += 1

                # Skip frames for performance
                if frame_count % frame_skip == 0:
                    # Update instance settings temporarily
                    self.hsv_lower = np.array([0, 0, v_min])
                    self.hsv_upper = np.array([180, s_max, 255])
                    self.min_area = max(1, min_area)
                    self.edge_expansion_pixels = max(1, edge_expand)
                    self.gradient_threshold = max(1, gradient_th)

                    # Use enhanced detection on display image
                    mask, _ = self.detect_glare_enhanced(
                        img_display, return_soft=False)

                    # Create visualization
                    overlay = img_display.copy()
                    overlay[mask > 0] = [0, 255, 255]  # Cyan for glare
                    blended = cv2.addWeighted(img_display, 0.65, overlay, 0.35, 0)

                    # Convert mask to color
                    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                    # Create display with all three views
                    display = np.hstack([img_display, mask_colored, blended])

                    # Add statistics and instructions
                    glare_pixels = np.sum(mask > 0)
                    total_pixels = mask.shape[0] * mask.shape[1]
                    coverage = (glare_pixels / total_pixels) * 100
                    num_labels, _, _, _ = cv2.connectedComponentsWithStats(
                        mask, connectivity=8)
                    num_regions = num_labels - 1

                    # Top text with stats
                    stats_text = f"Coverage: {coverage:.1f}%  |  Regions: {num_regions}  |  V_min: {v_min}  |  S_max: {s_max}  |  Edge: {edge_expand}  |  Grad: {gradient_th}"
                    cv2.putText(display, stats_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    # Bottom instructions
                    instructions = "Press: Q=Save&Quit | ESC=Quit | S=Save mask | H=Help"
                    cv2.putText(display, instructions, (10, display.shape[0]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    # Add labels for each view
                    label_y = 50
                    cv2.putText(display, "ORIGINAL", (img_display.shape[1]//2-50, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display, "MASK", (img_display.shape[1] + img_display.shape[1]//2-30, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display, "OVERLAY", (2*img_display.shape[1] + img_display.shape[1]//2-50, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow(window_name, display)
                    last_params = current_params

            # Wait for key press with longer delay to reduce CPU usage
            key = cv2.waitKey(50) & 0xFF

            # Check for quit keys
            if key == ord('q') or key == ord('Q'):
                # SAVE the settings permanently
                self.save_settings()
                print(f"\n✅ Tuning complete! Settings saved to {self.config_file}")
                break
            elif key == 27:  # ESC key
                print(f"\n⚠️  Tuning cancelled. Settings NOT saved.")
                # Reload original settings
                self.load_settings()
                break
            elif key == ord('s') or key == ord('S'):
                # Save current mask
                saved_count += 1
                filename = f'tuned_mask_{saved_count}.png'
                # Generate mask on ORIGINAL image, not display image
                temp_mask, _ = self.detect_glare_enhanced(image, return_soft=False)
                cv2.imwrite(filename, temp_mask)
                print(f"💾 Saved mask #{saved_count} to '{filename}'")
            elif key == ord('h') or key == ord('H'):
                # Help
                print("\n" + "="*60)
                print(" KEYBOARD SHORTCUTS ")
                print("="*60)
                print("  Q or q  = Quit and save settings")
                print("  ESC     = Quit without saving")
                print("  S or s  = Save current mask as PNG")
                print("  H or h  = Show this help")
                print("="*60 + "\n")

        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Final cleanup


# ========== TEST & DEMO FUNCTIONS ==========

def create_test_image():
    """Create a synthetic test image with artificial glare"""
    print("Creating synthetic test image...")

    # Create gray background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 120

    # Add text
    for i in range(8):
        cv2.putText(img, f"Sample Document Text Line {i+1}",
                    (50, 80 + i*60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Add glare spots with gradual edges
    centers = [(400, 200), (600, 400)]
    for center in centers:
        for radius in range(80, 10, -5):
            intensity = int(255 - (80 - radius) * 2)
            cv2.circle(img, center, radius,
                       (intensity, intensity, intensity), -1)

    # Add Gaussian blur
    mask_glare = np.zeros((600, 800), dtype=np.uint8)
    for center in centers:
        cv2.circle(mask_glare, center, 80, 255, -1)

    img_float = img.astype(float)
    for c in range(3):
        blurred = cv2.GaussianBlur(img[:, :, c].astype(float), (21, 21), 0)
        img_float[:, :, c] = np.where(
            mask_glare > 0, blurred, img_float[:, :, c])

    img = img_float.astype(np.uint8)

    cv2.imwrite('test_synthetic.jpg', img)
    print("✅ Created 'test_synthetic.jpg'")
    return img


def test_basic_functionality():
    """Test the module with basic cases"""
    print("\n" + "="*60)
    print("RUNNING BASIC TESTS")
    print("="*60)

    detector = EnhancedGlareDetector()

    # Test 1: Pure white spot
    print("\nTest 1: Pure white spot (should detect)")
    test1 = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(test1, (300, 200), 50, (255, 255, 255), -1)
    mask1 = detector.detect_glare_basic(test1)
    result1 = np.sum(mask1 > 0)
    print(f"  Detected pixels: {result1}")
    print(f"  ✅ PASS" if result1 > 1000 else f"  ❌ FAIL")

    # Test 2: Black image
    print("\nTest 2: Black image (should detect nothing)")
    test2 = np.zeros((400, 600, 3), dtype=np.uint8)
    mask2 = detector.detect_glare_basic(test2)
    result2 = np.sum(mask2 > 0)
    print(f"  Detected pixels: {result2}")
    print(f"  ✅ PASS" if result2 == 0 else f"  ❌ FAIL")

    # Test 3: Gray image
    print("\nTest 3: Gray image (should detect nothing)")
    test3 = np.ones((400, 600, 3), dtype=np.uint8) * 128
    mask3 = detector.detect_glare_basic(test3)
    result3 = np.sum(mask3 > 0)
    print(f"  Detected pixels: {result3}")
    print(f"  ✅ PASS" if result3 == 0 else f"  ❌ FAIL")

    print("\n" + "="*60)
    print("BASIC TESTS COMPLETE")
    print("="*60)


def demo_comparison():
    """Demonstrate the difference between basic and enhanced detection"""
    print("\n" + "="*60)
    print(" ENHANCED vs BASIC GLARE DETECTION COMPARISON ")
    print("="*60)

    # Load test image
    print("\n🔍 Attempting to load test image...")
    image = None
    test_names = [
        'test_image.jpg', 'test_image.JPG', 'test_image.jpeg',
        'test_image.JPEG', 'test_image.png', 'test_image.PNG',
    ]

    for name in test_names:
        if os.path.exists(name):
            image = cv2.imread(name)
            if image is not None:
                print(f"✅ Successfully loaded: {name}")
                break

    if image is None:
        print("⚠️  No test image found. Creating synthetic image...")
        image = create_test_image()

    if image is None:
        print("❌ CRITICAL ERROR: Could not create/load image")
        return None, None

    print(f"\n📊 Image loaded: {image.shape}")

    # Initialize detector
    detector = EnhancedGlareDetector()

    # Basic detection
    print("\n1️⃣  Running BASIC detection...")
    basic_mask = detector.detect_glare_basic(image, method='hsv')

    # Enhanced detection
    print("2️⃣  Running ENHANCED detection with all edge improvements...")
    enhanced_mask, soft_mask = detector.detect_glare_enhanced(image)

    # Calculate improvement
    basic_pixels = np.sum(basic_mask > 0)
    enhanced_pixels = np.sum(enhanced_mask > 0)
    improvement = ((enhanced_pixels - basic_pixels) /
                   basic_pixels * 100) if basic_pixels > 0 else 0

    print(f"\n📊 Comparison Results:")
    print(f"  Basic detection:    {basic_pixels:,} pixels")
    print(f"  Enhanced detection: {enhanced_pixels:,} pixels")
    print(f"  Improvement:        {improvement:+.1f}% coverage")

    # Get detailed statistics
    basic_stats = detector.get_statistics(basic_mask)
    enhanced_stats = detector.get_statistics(enhanced_mask)

    print(f"\n📈 Detailed Statistics:")
    print(
        f"  Basic    - Regions: {basic_stats['num_regions']}, Coverage: {basic_stats['coverage_percent']:.2f}%")
    print(
        f"  Enhanced - Regions: {enhanced_stats['num_regions']}, Coverage: {enhanced_stats['coverage_percent']:.2f}%")

    # Visualize comparison
    print("\n🎨 Generating comparison visualization...")
    detector.visualize_comparison(image, basic_mask, enhanced_mask, soft_mask,
                                  save_path='comparison_visualization.png')

    print("\n✅ Comparison demo complete!")
    return detector, image


def main():
    """Main demo function"""
    print("\n" + "="*70)
    print(" ENHANCED GLARE DETECTION MODULE - FULL DEMO ")
    print("="*70)

    # Run basic tests first
    test_basic_functionality()

    # Run comparison demo
    detector, image = demo_comparison()

    if detector is None or image is None:
        return

    # Interactive tuning
    print("\n" + "="*70)
    print("INTERACTIVE TUNING")
    print("="*70)
    print("\n🔧 Launching interactive tuning tool...")
    print("   Adjust the sliders to tune detection")
    print("   Press 'q' when done to SAVE your settings")
    print("   Press ESC to skip tuning")

    # Give user option to skip tuning
    print("\nPress ENTER to start tuning, or type 'skip' to skip: ", end='')
    user_input = input().strip().lower()

    if user_input != 'skip':
        detector.tune_thresholds(image)

        # Reload settings after tuning
        print("\n🔄 Reloading your saved settings...")
        detector.load_settings()

        # Run detection again with new settings
        print("\n📊 Running final detection with your saved settings...")
        final_mask, _ = detector.detect_glare_enhanced(image)

        cv2.imwrite('output_final_mask.png', final_mask)

        overlay_final = image.copy()
        overlay_final[final_mask > 0] = [0, 255, 255]  # Cyan
        result_final = cv2.addWeighted(image, 0.7, overlay_final, 0.3, 0)
        cv2.imwrite('output_final_overlay.png', result_final)

        print("\n💾 Saved final outputs:")
        print("  - output_final_mask.png")
        print("  - output_final_overlay.png")

        # Final visualization
        print("\n🎨 Generating final visualization...")
        detector.visualize(image, final_mask,
                           save_path='final_visualization.png')
    else:
        print("\n⏭️  Skipping interactive tuning")

    print("\n" + "="*70)
    print(" DEMO COMPLETE! ")
    print("="*70)
    print("\n📁 Generated files:")
    print("  Visualization:")
    print("    - comparison_visualization.png")
    if user_input != 'skip':
        print("    - final_visualization.png")
        print("  Final Detection:")
        print("    - output_final_mask.png")
        print("    - output_final_overlay.png")
    print("  Configuration:")
    print("    - glare_config.json (your saved settings)")

    print("\n✨ Key Features:")
    print("  ✓ Gradient-based edge detection")
    print("  ✓ Adaptive edge expansion")
    print("  ✓ Bloom/halo effect detection")
    print("  ✓ Morphological edge refinement")
    print("  ✓ Soft-edge feathering")
    print("  ✓ Settings save/load")
    print("  ✓ Interactive tuning")

    print("\n🚀 Usage Example:")
    print("  from enhanced_glare_detector import EnhancedGlareDetector")
    print("  ")
    print("  detector = EnhancedGlareDetector()")
    print("  mask, soft_mask = detector.detect_glare_enhanced(image)")
    print("  # Use soft_mask for better inpainting results!")


if __name__ == "__main__":
    main()