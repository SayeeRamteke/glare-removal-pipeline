"""
Module 5: Evaluation & Metrics
FIXED VERSION - Properly handles directory creation before saving

Quantifies improvement and compares against baselines
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os
from pathlib import Path


class EvaluationModule:
    """
    Evaluates glare removal quality by comparing against baselines
    and computing standard image quality metrics.
    """
    
    def __init__(self):
        self.baseline_single = None
        self.baseline_average = None
        self.reference_image = None  # Ground truth if available
        self.metrics = {}
        
    def create_baselines(self, img1: np.ndarray, img2_aligned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create baseline comparison methods.
        
        Args:
            img1: First image (with glare)
            img2_aligned: Second image aligned to img1
            
        Returns:
            baseline_single: Just the first image (do nothing)
            baseline_average: Simple 50-50 average of both images
        """
        # Baseline 1: Single image - just use first image as-is
        self.baseline_single = img1.copy()
        
        # Baseline 2: Simple averaging
        self.baseline_average = cv2.addWeighted(img1, 0.5, img2_aligned, 0.5, 0)
        
        return self.baseline_single, self.baseline_average
    
    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Higher values = more similar images = better quality.
        
        Typical values:
        - 20-25 dB: Poor quality
        - 25-30 dB: Acceptable
        - 30-40 dB: Good quality
        - 40+ dB: Excellent quality
        """
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index.
        
        Range: -1 to 1 (typically 0 to 1)
        - 1.0 = identical images
        - 0.9+ = excellent similarity
        - 0.8-0.9 = good similarity
        - Below 0.8 = noticeable differences
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        score = structural_similarity(gray1, gray2, data_range=255)
        
        return score
    
    def glare_statistics(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics about glare coverage.
        """
        glare_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage_percent = (glare_pixels / total_pixels) * 100
        
        num_labels, labels = cv2.connectedComponents(mask)
        num_glare_regions = num_labels - 1
        
        if num_glare_regions > 0:
            avg_region_size = glare_pixels / num_glare_regions
        else:
            avg_region_size = 0
        
        return {
            'glare_pixels': int(glare_pixels),
            'total_pixels': int(total_pixels),
            'coverage_percent': round(coverage_percent, 2),
            'num_regions': int(num_glare_regions),
            'avg_region_size': round(avg_region_size, 2)
        }
    
    def evaluate_result(self, 
                       img1: np.ndarray, 
                       img2_aligned: np.ndarray, 
                       result: np.ndarray,
                       mask1: np.ndarray,
                       mask2: np.ndarray,
                       reference: Optional[np.ndarray] = None) -> Dict:
        """
        Complete evaluation comparing our result against baselines.
        """
        # Create baselines
        baseline_single, baseline_average = self.create_baselines(img1, img2_aligned)
        
        # Get glare statistics
        glare_stats = self.glare_statistics(mask1)
        
        # If we have reference (ground truth), compare against it
        if reference is not None:
            self.reference_image = reference
            
            metrics = {
                'glare_stats': glare_stats,
                'vs_reference': {
                    'our_method': {
                        'psnr': self.compute_psnr(result, reference),
                        'ssim': self.compute_ssim(result, reference)
                    },
                    'baseline_single': {
                        'psnr': self.compute_psnr(baseline_single, reference),
                        'ssim': self.compute_ssim(baseline_single, reference)
                    },
                    'baseline_average': {
                        'psnr': self.compute_psnr(baseline_average, reference),
                        'ssim': self.compute_ssim(baseline_average, reference)
                    }
                }
            }
        else:
            # No reference available
            metrics = {
                'glare_stats': glare_stats,
                'vs_img1': {
                    'our_method': {
                        'psnr': self.compute_psnr(result, img1),
                        'ssim': self.compute_ssim(result, img1)
                    },
                    'baseline_average': {
                        'psnr': self.compute_psnr(baseline_average, img1),
                        'ssim': self.compute_ssim(baseline_average, img1)
                    }
                },
                'vs_img2': {
                    'our_method': {
                        'psnr': self.compute_psnr(result, img2_aligned),
                        'ssim': self.compute_ssim(result, img2_aligned)
                    },
                    'baseline_average': {
                        'psnr': self.compute_psnr(baseline_average, img2_aligned),
                        'ssim': self.compute_ssim(baseline_average, img2_aligned)
                    }
                }
            }
        
        # Calculate improvement percentages
        if reference is not None:
            psnr_improvement = ((metrics['vs_reference']['our_method']['psnr'] - 
                               metrics['vs_reference']['baseline_single']['psnr']) / 
                              metrics['vs_reference']['baseline_single']['psnr'] * 100)
            
            metrics['improvement_percent'] = {
                'psnr': round(psnr_improvement, 2)
            }
        
        self.metrics = metrics
        return metrics, self.baseline_average
    
    def print_metrics_report(self, metrics: Optional[Dict] = None):
        """Print a formatted report of evaluation metrics."""
        if metrics is None:
            metrics = self.metrics
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        # Glare statistics
        print("\nGLARE STATISTICS:")
        print("-" * 40)
        stats = metrics['glare_stats']
        print(f"Coverage: {stats['coverage_percent']}% of image")
        print(f"Total glare pixels: {stats['glare_pixels']:,}")
        print(f"Number of glare regions: {stats['num_regions']}")
        print(f"Average region size: {stats['avg_region_size']:.0f} pixels")
        
        # Quality metrics
        print("\nQUALITY METRICS:")
        print("-" * 40)
        
        if 'vs_reference' in metrics:
            # We have ground truth
            print("\nComparison against Reference (Ground Truth):")
            print(f"\n{'Method':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
            print("-" * 42)
            
            our = metrics['vs_reference']['our_method']
            base_single = metrics['vs_reference']['baseline_single']
            base_avg = metrics['vs_reference']['baseline_average']
            
            print(f"{'Our Method':<20} {our['psnr']:>8.2f}     {our['ssim']:>6.4f}")
            print(f"{'Baseline (Single)':<20} {base_single['psnr']:>8.2f}     {base_single['ssim']:>6.4f}")
            print(f"{'Baseline (Average)':<20} {base_avg['psnr']:>8.2f}     {base_avg['ssim']:>6.4f}")
            
            if 'improvement_percent' in metrics:
                print(f"\n✓ Improvement over single image: {metrics['improvement_percent']['psnr']:.1f}%")
        else:
            # No ground truth
            print("\nComparison (no ground truth available):")
            print("Note: Lower PSNR means more change from original")
            print(f"\n{'Comparison':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
            print("-" * 42)
            
            print("\nVs Image 1:")
            print(f"  {'Our Method':<18} {metrics['vs_img1']['our_method']['psnr']:>8.2f}     {metrics['vs_img1']['our_method']['ssim']:>6.4f}")
            print(f"  {'Average':<18} {metrics['vs_img1']['baseline_average']['psnr']:>8.2f}     {metrics['vs_img1']['baseline_average']['ssim']:>6.4f}")
            
            print("\nVs Image 2:")
            print(f"  {'Our Method':<18} {metrics['vs_img2']['our_method']['psnr']:>8.2f}     {metrics['vs_img2']['our_method']['ssim']:>6.4f}")
            print(f"  {'Average':<18} {metrics['vs_img2']['baseline_average']['psnr']:>8.2f}     {metrics['vs_img2']['baseline_average']['ssim']:>6.4f}")
        
        print("\n" + "="*60 + "\n")
    
    def generate_comparison_figure(self,
                                   img1: np.ndarray,
                                   img2_aligned: np.ndarray,
                                   result: np.ndarray,
                                   mask1: np.ndarray,
                                   mask2: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visual comparison figure showing all methods.
        FIXED: Properly creates directory before saving
        """
        # Create baselines
        baseline_single, baseline_average = self.create_baselines(img1, img2_aligned)
        
        # Create figure with 2 rows, 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Glare Removal Evaluation', fontsize=16, fontweight='bold')
        
        # Convert BGR to RGB for matplotlib
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)
        baseline_single_rgb = cv2.cvtColor(baseline_single, cv2.COLOR_BGR2RGB)
        baseline_avg_rgb = cv2.cvtColor(baseline_average, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Row 1: Input images and masks
        axes[0, 0].imshow(img1_rgb)
        axes[0, 0].set_title('Input Image 1\n(with glare)', fontsize=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_rgb)
        axes[0, 1].set_title('Input Image 2\n(aligned, different glare)', fontsize=10)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask1, cmap='hot')
        axes[0, 2].set_title('Detected Glare Mask', fontsize=10)
        axes[0, 2].axis('off')
        
        # Row 2: Baselines and result
        axes[1, 0].imshow(baseline_single_rgb)
        axes[1, 0].set_title('Baseline: Single Image\n(do nothing)', fontsize=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(baseline_avg_rgb)
        axes[1, 1].set_title('Baseline: Simple Average\n(50-50 blend)', fontsize=10)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(result_rgb)
        axes[1, 2].set_title('Our Result\n(Intelligent Reconstruction)', fontsize=10, 
                            fontweight='bold', color='darkgreen')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided - WITH DIRECTORY CHECK
        if save_path:
            try:
                # CRITICAL FIX: Create parent directory if it doesn't exist
                # This is the only line you need for this.
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
                # Now save the figure
                plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
                print(f"✓ Comparison figure saved to: {save_path}")

            except Exception as e:
                print(f"⚠️  Warning: Could not save comparison figure to {save_path}")
                print(f"   Error: {e}")

                plt.close(fig)  # Close to free memory
            return fig
    
    def save_metrics_to_file(self, filepath: str, metrics: Optional[Dict] = None):
        """
        Save metrics to a text file for record keeping.
        FIXED: Creates directory before saving
        """
        if metrics is None:
            metrics = self.metrics
        
        try:
            # CRITICAL FIX: Create parent directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write("GLARE REMOVAL EVALUATION METRICS\n")
                f.write("=" * 60 + "\n\n")
                
                # Glare stats
                f.write("GLARE STATISTICS:\n")
                f.write("-" * 40 + "\n")
                stats = metrics['glare_stats']
                f.write(f"Coverage: {stats['coverage_percent']}%\n")
                f.write(f"Glare pixels: {stats['glare_pixels']}\n")
                f.write(f"Number of regions: {stats['num_regions']}\n")
                f.write(f"Avg region size: {stats['avg_region_size']:.2f} pixels\n\n")
                
                # Quality metrics
                f.write("QUALITY METRICS:\n")
                f.write("-" * 40 + "\n")
                
                if 'vs_reference' in metrics:
                    f.write("\nVs Reference (Ground Truth):\n")
                    f.write(f"Our Method - PSNR: {metrics['vs_reference']['our_method']['psnr']:.2f} dB, "
                           f"SSIM: {metrics['vs_reference']['our_method']['ssim']:.4f}\n")
                    f.write(f"Baseline (Single) - PSNR: {metrics['vs_reference']['baseline_single']['psnr']:.2f} dB, "
                           f"SSIM: {metrics['vs_reference']['baseline_single']['ssim']:.4f}\n")
                    f.write(f"Baseline (Average) - PSNR: {metrics['vs_reference']['baseline_average']['psnr']:.2f} dB, "
                           f"SSIM: {metrics['vs_reference']['baseline_average']['ssim']:.4f}\n")
                else:
                    f.write("\nVs Image 1:\n")
                    f.write(f"Our Method - PSNR: {metrics['vs_img1']['our_method']['psnr']:.2f} dB, "
                           f"SSIM: {metrics['vs_img1']['our_method']['ssim']:.4f}\n")
                    f.write(f"Average - PSNR: {metrics['vs_img1']['baseline_average']['psnr']:.2f} dB, "
                           f"SSIM: {metrics['vs_img1']['baseline_average']['ssim']:.4f}\n")
                    
                    f.write("\nVs Image 2:\n")
                    f.write(f"Our Method - PSNR: {metrics['vs_img2']['our_method']['psnr']:.2f} dB, "
                           f"SSIM: {metrics['vs_img2']['our_method']['ssim']:.4f}\n")
                    f.write(f"Average - PSNR: {metrics['vs_img2']['baseline_average']['psnr']:.2f} dB, "
                           f"SSIM: {metrics['vs_img2']['baseline_average']['ssim']:.4f}\n")
            
            print(f"✓ Metrics saved to: {filepath}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save metrics to {filepath}")
            print(f"   Error: {e}")


# Example usage and testing
if __name__ == "__main__":
    """Test the evaluation module with dummy images."""
    print("Testing Evaluation Module...")
    
    # Create dummy test images
    img1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
    img2 = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # Add fake glare to img1
    cv2.circle(img1, (320, 240), 80, (255, 255, 255), -1)
    
    # Create fake glare mask
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask1, (320, 240), 80, 255, -1)
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    
    # Simulate result (glare removed)
    result = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # Initialize evaluation module
    evaluator = EvaluationModule()
    
    # Run evaluation
    metrics = evaluator.evaluate_result(img1, img2, result, mask1, mask2)
    
    # Print report
    evaluator.print_metrics_report()
    
    # Generate comparison figure
    fig = evaluator.generate_comparison_figure(img1, img2, result, mask1, mask2, 
                                               save_path='test_comparison.png')
    
    print("\n✓ Module 5 test complete!")
    print("Check 'test_comparison.png' for visual output")