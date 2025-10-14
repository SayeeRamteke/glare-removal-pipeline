"""
Test script for Image Reconstruction Module
Purpose: Test reconstruction module with custom image pairs and masks

Usage:
    python tests/test_reconstruction.py

Structure:
    Place your test images in: IP_PROJECT/data/test_inputs/
    Results will be saved in: IP_PROJECT/data/test_results/
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from modules.reconstruction_module import ReconstructionModule


class ReconstructionTester:
    def __init__(self, data_dir='data'):
        """Initialize tester with data directory."""
        self.data_dir = Path(data_dir)
        self.input_dir = self.data_dir / 'test_inputs'
        self.results_dir = self.data_dir / 'test_results'
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Test directories ready:")
        print(f"  Input:   {self.input_dir}")
        print(f"  Results: {self.results_dir}")
    
    def load_test_pair(self, img1_path, img2_path, mask1_path, mask2_path):
        """
        Load a pair of images and their masks.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second (aligned) image
            mask1_path: Path to mask for first image
            mask2_path: Path to mask for second image
            
        Returns:
            Tuple of (img1, img2, mask1, mask2) or None if loading fails
        """
        try:
            # Load images
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            # Load masks (grayscale)
            mask1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(str(mask2_path), cv2.IMREAD_GRAYSCALE)
            
            # Validate
            if img1 is None or img2 is None:
                print(f"✗ Error: Could not load images")
                return None
            
            if mask1 is None or mask2 is None:
                print(f"✗ Error: Could not load masks")
                return None
            
            print(f"✓ Loaded images: {img1.shape}, {img2.shape}")
            print(f"✓ Loaded masks: {mask1.shape}, {mask2.shape}")
            
            return img1, img2, mask1, mask2
            
        except Exception as e:
            print(f"✗ Error loading test pair: {e}")
            return None
    
    def run_reconstruction_test(self, img1, img2_aligned, mask1, mask2, 
                               test_name, blend_method='blended'):
        """
        Run reconstruction and save results.
        
        Args:
            img1: First image
            img2_aligned: Second aligned image
            mask1: Mask for first image
            mask2: Mask for second image
            test_name: Name for this test (used in output files)
            blend_method: 'simple' or 'blended'
            
        Returns:
            Dictionary with results and paths
        """
        print(f"\n{'='*60}")
        print(f"Running Test: {test_name}")
        print(f"Method: {blend_method}")
        print(f"{'='*60}")
        
        # Initialize reconstruction module
        reconstructor = ReconstructionModule(blend_method=blend_method)
        
        # Run reconstruction
        result, overlap_mask, stats = reconstructor.reconstruct(
            img1, img2_aligned, mask1, mask2
        )
        
        # Create visualization
        visualization = reconstructor.visualize_reconstruction(
            img1, img2_aligned, result, mask1, mask2, overlap_mask
        )
        
        # Find next available test number
        test_num = self._get_next_test_number(test_name)
        test_id = f"{test_name}_{test_num}"
        
        # Save results
        output_dir = self.results_dir / test_id
        output_dir.mkdir(exist_ok=True)
        
        # Save individual outputs
        result_path = output_dir / 'result.png'
        overlap_path = output_dir / 'overlap_mask.png'
        vis_path = output_dir / 'visualization.png'
        
        cv2.imwrite(str(result_path), result)
        cv2.imwrite(str(overlap_path), overlap_mask)
        cv2.imwrite(str(vis_path), visualization)
        
        # Save statistics
        stats_path = output_dir / 'stats.txt'
        self._save_statistics(stats_path, stats, blend_method)
        
        # Copy input files for reference
        cv2.imwrite(str(output_dir / 'input_img1.png'), img1)
        cv2.imwrite(str(output_dir / 'input_img2.png'), img2_aligned)
        cv2.imwrite(str(output_dir / 'input_mask1.png'), mask1)
        cv2.imwrite(str(output_dir / 'input_mask2.png'), mask2)
        
        # Print results
        print(f"\n📊 Reconstruction Statistics:")
        print(f"  Glare pixels (img1): {stats['glare1_pixels']:,}")
        print(f"  Glare pixels (img2): {stats['glare2_pixels']:,}")
        print(f"  Overlapping glare:   {stats['overlap_pixels']:,} ({stats['overlap_percentage']:.1f}%)")
        print(f"  Recoverable:         {stats['recoverable_percentage']:.1f}%")
        
        print(f"\n💾 Results saved to: {output_dir}")
        print(f"  - result.png          : Final reconstructed image")
        print(f"  - overlap_mask.png    : Overlapping glare regions")
        print(f"  - visualization.png   : Complete comparison view")
        print(f"  - stats.txt           : Detailed statistics")
        
        return {
            'test_id': test_id,
            'result': result,
            'overlap_mask': overlap_mask,
            'stats': stats,
            'paths': {
                'result': result_path,
                'overlap': overlap_path,
                'visualization': vis_path,
                'stats': stats_path
            }
        }
    
    def _get_next_test_number(self, test_name):
        """Find the next available test number for a given test name."""
        existing = list(self.results_dir.glob(f"{test_name}_*"))
        if not existing:
            return 1
        
        # Extract numbers from existing folders
        numbers = []
        for folder in existing:
            try:
                num = int(folder.name.split('_')[-1])
                numbers.append(num)
            except:
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def _save_statistics(self, path, stats, blend_method):
        """Save detailed statistics to text file."""
        with open(path, 'w') as f:
            f.write("Image Reconstruction Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Blend Method: {blend_method}\n\n")
            
            f.write("Glare Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Image 1 glare pixels:     {stats['glare1_pixels']:,}\n")
            f.write(f"Image 2 glare pixels:     {stats['glare2_pixels']:,}\n")
            f.write(f"Overlapping glare pixels: {stats['overlap_pixels']:,}\n\n")
            
            f.write("Recovery Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Overlap percentage:    {stats['overlap_percentage']:.2f}%\n")
            f.write(f"Recoverable glare:     {stats['recoverable_percentage']:.2f}%\n\n")
            
            if stats['overlap_percentage'] > 50:
                f.write("\n⚠️  WARNING: High overlap detected!\n")
                f.write("Consider capturing a third image from a different angle.\n")
    
    def batch_test(self, test_configs):
        """
        Run multiple tests in batch.
        
        Args:
            test_configs: List of dictionaries with keys:
                - 'img1': path to first image
                - 'img2': path to second image
                - 'mask1': path to first mask
                - 'mask2': path to second mask
                - 'name': test name
                - 'method': (optional) blend method
        
        Returns:
            List of results for each test
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"BATCH TEST: Running {len(test_configs)} tests")
        print(f"{'='*60}\n")
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n[{i}/{len(test_configs)}] Testing: {config['name']}")
            
            # Load images
            data = self.load_test_pair(
                config['img1'], config['img2'],
                config['mask1'], config['mask2']
            )
            
            if data is None:
                print(f"✗ Skipping test {config['name']} due to loading error")
                continue
            
            img1, img2, mask1, mask2 = data
            
            # Run test
            method = config.get('method', 'blended')
            result = self.run_reconstruction_test(
                img1, img2, mask1, mask2,
                config['name'], method
            )
            
            results.append(result)
        
        print(f"\n{'='*60}")
        print(f"✓ Batch test complete: {len(results)}/{len(test_configs)} successful")
        print(f"{'='*60}\n")
        
        return results


def main():
    """Main test function with example usage."""
    print("\n" + "="*60)
    print("Image Reconstruction Module - Test Suite")
    print("="*60 + "\n")
    
    # Initialize tester
    tester = ReconstructionTester()
    
    print("\n📝 Instructions:")
    print("  1. Place your test images in: data/test_inputs/")
    print("  2. Name your files as:")
    print("     - img1_testname.png")
    print("     - img2_testname.png")
    print("     - mask1_testname.png")
    print("     - mask2_testname.png")
    print("\n")
    
    # Example: Manual single test
    print("Example 1: Single Test")
    print("-" * 60)
    
    # Define paths (update these with your actual file names)
    test_name = "test1"
    img1_path = tester.input_dir / f'img1_{test_name}.png'
    img2_path = tester.input_dir / f'img2_{test_name}.png'
    mask1_path = tester.input_dir / f'mask1_{test_name}.png'
    mask2_path = tester.input_dir / f'mask2_{test_name}.png'
    
    # Check if files exist
    if all(p.exists() for p in [img1_path, img2_path, mask1_path, mask2_path]):
        data = tester.load_test_pair(img1_path, img2_path, mask1_path, mask2_path)
        if data:
            img1, img2, mask1, mask2 = data
            result = tester.run_reconstruction_test(
                img1, img2, mask1, mask2,
                test_name='testresult',
                blend_method='blended'
            )
    else:
        print("ℹ️  Test files not found. Please add your test images first.")
        print(f"   Looking for files in: {tester.input_dir}")
    
    print("\n" + "="*60)
    print("\nExample 2: Batch Test")
    print("-" * 60)
    
    # Example batch configuration
    batch_configs = [
        {
            'img1': tester.input_dir / 'img1_test1.png',
            'img2': tester.input_dir / 'img2_test1.png',
            'mask1': tester.input_dir / 'mask1_test1.png',
            'mask2': tester.input_dir / 'mask2_test1.png',
            'name': 'testresult',
            'method': 'blended'
        },
        {
            'img1': tester.input_dir / 'img1_test2.png',
            'img2': tester.input_dir / 'img2_test2.png',
            'mask1': tester.input_dir / 'mask1_test2.png',
            'mask2': tester.input_dir / 'mask2_test2.png',
            'name': 'testresult',
            'method': 'simple'
        }
    ]
    
    # Check if any batch files exist
    existing_configs = [c for c in batch_configs 
                       if all(Path(c[k]).exists() for k in ['img1', 'img2', 'mask1', 'mask2'])]
    
    if existing_configs:
        results = tester.batch_test(existing_configs)
        print(f"\n✓ Completed {len(results)} tests")
    else:
        print("ℹ️  No batch test files found.")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()