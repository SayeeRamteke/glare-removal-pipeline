"""
Test script for Reconstruction Module
Tests the module independently by simulating glare masks
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path to import reconstruction module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.reconstruction_module import ReconstructionModule


class MaskSimulator:
    """Simulate glare masks for testing purposes."""
    
    @staticmethod
    def create_circular_glare(img_shape, center, radius, intensity=255):
        """
        Create a circular glare mask.
        
        Args:
            img_shape: (height, width) of image
            center: (x, y) center of glare
            radius: radius of glare circle
            intensity: brightness value (default 255)
            
        Returns:
            Binary mask with circular glare
        """
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        cv2.circle(mask, center, radius, intensity, -1)
        
        # Add some blur for realistic edge
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    @staticmethod
    def create_elliptical_glare(img_shape, center, axes, angle, intensity=255):
        """
        Create an elliptical glare mask.
        
        Args:
            img_shape: (height, width) of image
            center: (x, y) center of glare
            axes: (major_axis, minor_axis) lengths
            angle: rotation angle in degrees
            intensity: brightness value
            
        Returns:
            Binary mask with elliptical glare
        """
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, intensity, -1)
        
        # Add blur for realistic edge
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    @staticmethod
    def create_random_glare(img_shape, num_spots=3, size_range=(30, 80)):
        """
        Create random glare spots on image.
        
        Args:
            img_shape: (height, width) of image
            num_spots: number of glare spots
            size_range: (min_radius, max_radius) for spots
            
        Returns:
            Binary mask with random glare spots
        """
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        h, w = img_shape[:2]
        
        for _ in range(num_spots):
            # Random center (avoid edges)
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            
            # Random radius
            radius = np.random.randint(size_range[0], size_range[1])
            
            # Draw circle
            cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Blur and threshold
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        
        return mask
    
    @staticmethod
    def simulate_glare_on_image(img, mask, glare_intensity=0.8):
        """
        Apply simulated glare effect to an image.
        
        Args:
            img: Original image
            mask: Binary glare mask
            glare_intensity: How bright the glare should be (0-1)
            
        Returns:
            Image with simulated glare
        """
        result = img.copy().astype(np.float32)
        
        # Create glare effect (bright white overlay)
        glare_overlay = np.ones_like(result) * 255
        
        # Blend using mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        result = (1 - mask_3ch * glare_intensity) * result + (mask_3ch * glare_intensity) * glare_overlay
        
        return result.astype(np.uint8)


def test_basic_reconstruction():
    """Test basic reconstruction with simulated data."""
    print("\n" + "="*60)
    print("TEST 1: Basic Reconstruction with Simulated Masks")
    print("="*60)
    
    # Create synthetic test image (colorful pattern)
    img_height, img_width = 600, 800
    img_base = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Create a pattern (gradient + checkerboard)
    for i in range(img_height):
        for j in range(img_width):
            img_base[i, j] = [
                int(255 * i / img_height),  # B
                int(255 * j / img_width),    # G
                128                          # R
            ]
    
    # Add checkerboard pattern
    square_size = 50
    for i in range(0, img_height, square_size):
        for j in range(0, img_width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img_base[i:i+square_size, j:j+square_size] = [50, 50, 50]
    
    # Create two different glare masks
    mask1 = MaskSimulator.create_circular_glare(
        img_base.shape, center=(300, 250), radius=100
    )
    
    mask2 = MaskSimulator.create_circular_glare(
        img_base.shape, center=(500, 350), radius=120
    )
    
    # Simulate glare on images
    img1 = MaskSimulator.simulate_glare_on_image(img_base, mask1, 0.85)
    img2 = MaskSimulator.simulate_glare_on_image(img_base, mask2, 0.85)
    
    # Test both reconstruction methods
    for method in ['simple', 'blended']:
        print(f"\nTesting {method} reconstruction...")
        
        reconstructor = ReconstructionModule(blend_method=method)
        result, overlap, stats = reconstructor.reconstruct(img1, img2, mask1, mask2)
        
        print(f"✓ Reconstruction completed successfully!")
        print(f"  - Method: {stats['method_used']}")
        print(f"  - Glare in img1: {stats['glare1_pixels']} pixels")
        print(f"  - Glare in img2: {stats['glare2_pixels']} pixels")
        print(f"  - Overlap: {stats['overlap_pixels']} pixels ({stats['overlap_percentage']:.1f}%)")
        print(f"  - Recoverable: {stats['recoverable_percentage']:.1f}%")
        
        # Create visualization
        vis = reconstructor.visualize_reconstruction(
            img1, img2, result, mask1, mask2, overlap
        )
        
        # Save results
        output_dir = 'data/test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(f'{output_dir}/test1_{method}_result.jpg', result)
        cv2.imwrite(f'{output_dir}/test1_{method}_visualization.jpg', vis)
        print(f"  - Saved to: {output_dir}/test1_{method}_*")


def test_with_real_images():
    """Test reconstruction with real images from data folder."""
    print("\n" + "="*60)
    print("TEST 2: Reconstruction with Real Images")
    print("="*60)
    
    # Look for test images in data folder
    img1_path = 'data/image1.jpg'
    img2_path = 'data/image2.jpg'
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("\n⚠️  No real test images found in data/ folder")
        print("   To test with real images:")
        print("   1. Place two images with glare in data/ folder")
        print("   2. Name them 'image1.jpg' and 'image2.jpg'")
        print("   3. Run this test again")
        return
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("✗ Error loading images")
        return
    
    print(f"✓ Loaded images: {img1.shape}")
    
    # Simulate masks (since we don't have real glare detection yet)
    print("  Creating simulated glare masks...")
    h, w = img1.shape[:2]
    
    # Create realistic glare patterns
    mask1 = MaskSimulator.create_random_glare(img1.shape, num_spots=2)
    mask2 = MaskSimulator.create_random_glare(img2.shape, num_spots=2)
    
    # Test reconstruction
    reconstructor = ReconstructionModule(blend_method='blended')
    result, overlap, stats = reconstructor.reconstruct(img1, img2, mask1, mask2)
    
    print(f"\n✓ Reconstruction completed!")
    print(f"  - Overlap: {stats['overlap_percentage']:.1f}%")
    print(f"  - Recoverable: {stats['recoverable_percentage']:.1f}%")
    
    # Create and save visualization
    vis = reconstructor.visualize_reconstruction(img1, img2, result, mask1, mask2, overlap)
    
    output_dir = 'data/test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f'{output_dir}/real_images_result.jpg', result)
    cv2.imwrite(f'{output_dir}/real_images_visualization.jpg', vis)
    print(f"  - Saved to: {output_dir}/real_images_*")


def test_overlap_scenarios():
    """Test different overlap scenarios."""
    print("\n" + "="*60)
    print("TEST 3: Different Overlap Scenarios")
    print("="*60)
    
    # Create base image
    img_height, img_width = 400, 600
    img_base = np.random.randint(50, 200, (img_height, img_width, 3), dtype=np.uint8)
    
    scenarios = [
        ("No Overlap", (150, 200), (450, 200)),
        ("Partial Overlap", (250, 200), (350, 200)),
        ("High Overlap", (300, 200), (320, 200)),
    ]
    
    for scenario_name, center1, center2 in scenarios:
        print(f"\n{scenario_name}:")
        
        # Create masks
        mask1 = MaskSimulator.create_circular_glare(img_base.shape, center1, 80)
        mask2 = MaskSimulator.create_circular_glare(img_base.shape, center2, 80)
        
        # Simulate glare
        img1 = MaskSimulator.simulate_glare_on_image(img_base, mask1, 0.9)
        img2 = MaskSimulator.simulate_glare_on_image(img_base, mask2, 0.9)
        
        # Reconstruct
        reconstructor = ReconstructionModule(blend_method='blended')
        result, overlap, stats = reconstructor.reconstruct(img1, img2, mask1, mask2)
        
        print(f"  - Overlap: {stats['overlap_percentage']:.1f}%")
        print(f"  - Recoverable: {stats['recoverable_percentage']:.1f}%")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases and Error Handling")
    print("="*60)
    
    # Create test images
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    reconstructor = ReconstructionModule()
    
    # Test 1: Mismatched image sizes
    print("\n1. Testing mismatched image sizes...")
    img_wrong = np.zeros((100, 150, 3), dtype=np.uint8)
    try:
        reconstructor.reconstruct(img, img_wrong, mask, mask)
        print("  ✗ Should have raised error")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {str(e)[:50]}...")
    
    # Test 2: Wrong mask dimensions
    print("\n2. Testing wrong mask dimensions...")
    mask_wrong = np.zeros((50, 50), dtype=np.uint8)
    try:
        reconstructor.reconstruct(img, img, mask_wrong, mask)
        print("  ✗ Should have raised error")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {str(e)[:50]}...")
    
    # Test 3: No glare case
    print("\n3. Testing with no glare (empty masks)...")
    result, overlap, stats = reconstructor.reconstruct(img, img, mask, mask)
    print(f"  ✓ Handled gracefully: {stats['glare1_pixels']} glare pixels")
    
    # Test 4: Full image glare
    print("\n4. Testing with full image glare...")
    mask_full = np.ones((100, 100), dtype=np.uint8) * 255
    result, overlap, stats = reconstructor.reconstruct(img, img, mask_full, mask_full)
    print(f"  ✓ Handled gracefully: {stats['overlap_percentage']:.1f}% overlap")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("RECONSTRUCTION MODULE - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    try:
        test_basic_reconstruction()
        test_with_real_images()
        test_overlap_scenarios()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck 'data/test_results/' folder for output images")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/test_results', exist_ok=True)
    
    # Run all tests
    run_all_tests()