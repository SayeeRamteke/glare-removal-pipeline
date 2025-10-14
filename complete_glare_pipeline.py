"""
Complete Glare Removal Pipeline with Organized Folder Structure
FIXED VERSION - Properly creates all directories

Usage:
    python complete_glare_pipeline.py img1.jpg img2.jpg
    python complete_glare_pipeline.py img1.jpg img2.jpg --tune
"""

import cv2
import numpy as np
import sys
import os
from typing import Tuple, Dict, Optional
from pathlib import Path
import json
from datetime import datetime


from modules.enhanced_glare_detector import EnhancedGlareDetector
from modules.reconstruction_module import ReconstructionModule
from modules.evaluation_module import EvaluationModule

# Try to import alignment module
try:
    from modules.alignment_module import ImageAligner
    HAS_ALIGNMENT_MODULE = True
except ImportError:
    try:
        from modules.alignment_module import ImageAligner
        HAS_ALIGNMENT_MODULE = True
    except ImportError:
        HAS_ALIGNMENT_MODULE = False

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
class CompleteGlarePipeline:
    """Complete end-to-end pipeline with organized folder structure"""
    
    def __init__(self, 
                 input_dir: str = "data/inputs",
                 output_dir: str = "data/outputs", 
                 config_dir: str = "data/config",
                 blend_method: str = 'blended'):
        """Initialize pipeline with folder structure"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)
        
        # Create base directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modules
        config_file = self.config_dir / "glare_config.json"
        self.detector = EnhancedGlareDetector(config_file=str(config_file))
        self.reconstructor = ReconstructionModule(blend_method=blend_method)
        self.evaluator = EvaluationModule()
        
        # Alignment module
        if HAS_ALIGNMENT_MODULE:
            self.aligner = ImageAligner()
        else:
            self.aligner = None
        
        self.use_enhanced_detection = True
        self.target_resolution = None
        
        print(f"\n📁 Pipeline initialized with:")
        print(f"   Input:  {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Config: {self.config_dir}")
    
    def _save_to_step(self, step: str, filename: str, image: np.ndarray):
        """Helper to save images to appropriate step folder"""
        step_dir = self.run_output_dir / step
        step_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
        filepath = step_dir / filename
        cv2.imwrite(str(filepath), image)
        return filepath
    
    def resize_to_match(self, img1: np.ndarray, img2: np.ndarray, 
                       target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Resize both images to same size"""
        if target_size is None:
            target_size = (img1.shape[1], img1.shape[0])
        
        h, w = target_size[1], target_size[0]
        img1_resized = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LANCZOS4)
        img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        print(f"   Resized to: {w}x{h}")
        return img1_resized, img2_resized
    
    def process(self, 
                img1_filename: str, 
                img2_filename: str,
                tune_detection: bool = False,
                alignment_method: str = 'auto',
                target_resolution: Optional[str] = None) -> Dict:
        """Main pipeline execution"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique run directory
        self.run_output_dir = self.output_dir / f"RUN_{timestamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # PRE-CREATE ALL STEP DIRECTORIES
        for step in ['step1_load', 'step2_align', 'step3_detect', 'step4_reconstruct', 'step5_evaluate']:
            (self.run_output_dir / step).mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print(" COMPLETE GLARE REMOVAL PIPELINE ")
        print("="*70)
        print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Output: {self.run_output_dir.name}")
        print("="*70)
        
        # Parse target resolution
        if target_resolution:
            try:
                w, h = map(int, target_resolution.lower().split('x'))
                self.target_resolution = (w, h)
                print(f"\n🎯 Target resolution: {w}x{h}")
            except:
                print(f"\n⚠️  Invalid resolution: {target_resolution}")
                self.target_resolution = None
        
        # ==================== STEP 1: LOAD & RESIZE ====================
        print("\n" + "─"*70)
        print("📂 STEP 1/5: Loading Images")
        print("─"*70)
        
        img1_path = self.input_dir / img1_filename
        img2_path = self.input_dir / img2_filename
        
        if not img1_path.exists():
            raise FileNotFoundError(f"Image not found: {img1_path}")
        if not img2_path.exists():
            raise FileNotFoundError(f"Image not found: {img2_path}")
        
        img1_original = cv2.imread(str(img1_path))
        img2_original = cv2.imread(str(img2_path))
        
        if img1_original is None or img2_original is None:
            raise ValueError("Failed to load images")
        
        print(f"✓ Loaded: {img1_filename} → {img1_original.shape}")
        print(f"✓ Loaded: {img2_filename} → {img2_original.shape}")
        
        # Resize to same size
        if img1_original.shape != img2_original.shape or self.target_resolution:
            print("\n🔄 Resizing images to match...")
            img1, img2 = self.resize_to_match(img1_original, img2_original, 
                                              self.target_resolution)
        else:
            img1, img2 = img1_original, img2_original
            print("✓ Images already same size")
        
        self._save_to_step("step1_load", "1_img1_original.png", img1)
        self._save_to_step("step1_load", "2_img2_original.png", img2)
        print(f"✓ Final size: {img1.shape}")
        
        # ==================== STEP 2: ALIGNMENT ====================
        print("\n" + "─"*70)
        print("🔄 STEP 2/5: Image Alignment")
        print("─"*70)
        
        if HAS_ALIGNMENT_MODULE and self.aligner:
            print(f"Using alignment module with method: {alignment_method}")
            img2_aligned, alignment_info = self._align_with_module(img1, img2, alignment_method)
        else:
            print("Using fallback ORB+Homography alignment")
            img2_aligned, alignment_info = self._align_fallback(img1, img2)
        
        if img2_aligned is None:
            print("\n❌ ALIGNMENT FAILED!")
            return {'success': False, 'error': 'alignment_failed'}
        
        print(f"✓ Alignment successful!")
        print(f"  Method: {alignment_info['method']}")
        if 'confidence' in alignment_info:
            print(f"  Confidence: {alignment_info['confidence']:.1f}%")
        if 'matches' in alignment_info:
            print(f"  Matches: {alignment_info['matches']}")
        
        self._save_to_step("step2_align", "1_img2_aligned.png", img2_aligned)
        align_viz = self._create_alignment_viz(img1, img2, img2_aligned)
        self._save_to_step("step2_align", "2_alignment_comparison.png", align_viz)
        
        # ==================== STEP 3: GLARE DETECTION ====================
        print("\n" + "─"*70)
        print("🔍 STEP 3/5: Glare Detection")
        print("─"*70)
        
        if tune_detection:
            print("\n🔧 Launching interactive tuning...")
            self.detector.tune_thresholds(img1)
            print("✓ Settings saved")
        
        print(f"\nDetecting glare ({'ENHANCED' if self.use_enhanced_detection else 'BASIC'})...")
        
        if self.use_enhanced_detection:
            mask1, soft_mask1 = self.detector.detect_glare_enhanced(img1)
            mask2, soft_mask2 = self.detector.detect_glare_enhanced(img2_aligned)
        else:
            mask1 = self.detector.detect_glare_basic(img1)
            mask2 = self.detector.detect_glare_basic(img2_aligned)
        
        stats1 = self.detector.get_statistics(mask1)
        stats2 = self.detector.get_statistics(mask2)
        
        print(f"\n📊 Glare Detection Results:")
        print(f"   Image 1: {stats1['coverage_percent']:.2f}% coverage, {stats1['num_regions']} regions")
        print(f"   Image 2: {stats2['coverage_percent']:.2f}% coverage, {stats2['num_regions']} regions")
        
        self._save_to_step("step3_detect", "1_mask1.png", mask1)
        self._save_to_step("step3_detect", "2_mask2.png", mask2)
        
        overlay1 = self._create_overlay(img1, mask1)
        overlay2 = self._create_overlay(img2_aligned, mask2)
        self._save_to_step("step3_detect", "3_detection1_overlay.png", overlay1)
        self._save_to_step("step3_detect", "4_detection2_overlay.png", overlay2)
        
        detect_viz = self._create_detection_viz(img1, img2_aligned, mask1, mask2)
        self._save_to_step("step3_detect", "5_detection_comparison.png", detect_viz)
        
        # ==================== STEP 4: RECONSTRUCTION ====================
        print("\n" + "─"*70)
        print("🔨 STEP 4/5: Image Reconstruction")
        print("─"*70)
        
        print(f"Reconstructing using '{self.reconstructor.blend_method}' method...")
        
        result, overlap_mask, recon_stats = self.reconstructor.reconstruct(
            img1, img2_aligned, mask1, mask2
        )
        
        print(f"\n📊 Reconstruction Statistics:")
        print(f"   Overlap: {recon_stats['overlap_percentage']:.2f}%")
        print(f"   Recoverable: {recon_stats['recoverable_percentage']:.2f}%")
        
        if recon_stats['overlap_percentage'] < 10:
            quality = "🌟 EXCELLENT"
        elif recon_stats['overlap_percentage'] < 30:
            quality = "✓ GOOD"
        elif recon_stats['overlap_percentage'] < 50:
            quality = "⚠️  FAIR"
        else:
            quality = "❌ POOR - Consider more images"
        
        print(f"   Quality: {quality}")
        
        self._save_to_step("step4_reconstruct", "1_result.png", result)
        self._save_to_step("step4_reconstruct", "2_overlap_mask.png", overlap_mask)
        
        recon_viz = self.reconstructor.visualize_reconstruction(
            img1, img2_aligned, result, mask1, mask2, overlap_mask
        )
        self._save_to_step("step4_reconstruct", "3_reconstruction_viz.png", recon_viz)
        
        # ==================== STEP 5: EVALUATION ====================
        print("\n" + "─"*70)
        print("📊 STEP 5/5: Evaluation & Metrics")
        print("─"*70)
        
        print("Computing quality metrics...")
        
        eval_metrics,baseline_average_img = self.evaluator.evaluate_result(
            img1, img2_aligned, result, mask1, mask2, reference=None
        )
        
        self.evaluator.print_metrics_report(eval_metrics)
        
        # Save evaluation outputs - directory already exists
        eval_fig_path = self.run_output_dir / "step5_evaluate" / "1_method_comparison.png"
        print(f"\n💾 Saving evaluation figure to: {eval_fig_path}")
        
        try:
            eval_fig = self.evaluator.generate_comparison_figure(
                img1, img2_aligned, result, mask1, mask2,
                save_path=str(eval_fig_path)
            )
            print(f"✓ Saved: {eval_fig_path.name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save comparison figure: {e}")
        
        metrics_file = self.run_output_dir / "step5_evaluate" / "2_metrics_report.txt"
        try:
            self.evaluator.save_metrics_to_file(str(metrics_file), eval_metrics)
            print(f"✓ Saved: {metrics_file.name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save metrics file: {e}")
        
        try:
            avg_baseline_path = self.run_output_dir / "step5_evaluate" / "3_baseline_average.png"
            cv2.imwrite(str(avg_baseline_path), baseline_average_img)
            print(f"✓ Saved: {avg_baseline_path.name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save the average baseline image: {e}")
            
        # ==================== SAVE FINAL OUTPUTS ====================
        print("\n" + "─"*70)
        print("💾 Saving Final Outputs")
        print("─"*70)
        
        final_result_path = self.run_output_dir / "FINAL_result.png"
        final_comparison_path = self.run_output_dir / "FINAL_comparison.png"
        
        cv2.imwrite(str(final_result_path), result)
        
        final_comparison = self._create_final_comparison(img1, img2, result)
        cv2.imwrite(str(final_comparison_path), final_comparison)
        
        print(f"✓ {final_result_path.name}")
        print(f"✓ {final_comparison_path.name}")
        
        summary = self._create_summary(
            img1_filename, img2_filename, alignment_info, 
            stats1, stats2, recon_stats, eval_metrics, timestamp
        )
        summary_path = self.run_output_dir / "FINAL_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4,cls=NumpyEncoder)
        print(f"✓ {summary_path.name}")
        
        # ==================== COMPLETION ====================
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE! ")
        print("="*70)
        
        print(f"\n📁 Output Location: {self.run_output_dir}")
        print(f"\n🎯 Main Results:")
        print(f"   • FINAL_result.png")
        print(f"   • FINAL_comparison.png")
        print(f"   • FINAL_summary.json")
        print(f"\n📂 Step-by-step outputs in:")
        print(f"   • step1_load/")
        print(f"   • step2_align/")
        print(f"   • step3_detect/")
        print(f"   • step4_reconstruct/")
        print(f"   • step5_evaluate/")
        
        return {
            'success': True,
            'timestamp': timestamp,
            'result_image': result,
            'final_result_path': str(final_result_path),
            'final_comparison_path': str(final_comparison_path),
            'alignment': alignment_info,
            'detection': {'stats1': stats1, 'stats2': stats2},
            'reconstruction': recon_stats,
            'evaluation': eval_metrics
        }
    
    # ==================== ALIGNMENT ====================
    
    def _align_with_module(self, img1, img2, method):
        """Use alignment module if available"""
        try:
            if method == 'auto':
                aligned, info = self.aligner.align_auto(img2, img1)
            else:
                aligned, info = self.aligner.align(img2, img1, method=method)
            return aligned, info
        except Exception as e:
            print(f"⚠️  Alignment module error: {e}")
            return self._align_fallback(img1, img2)
    
    def _align_fallback(self, img1, img2):
        """Fallback ORB+Homography alignment"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None, {'method': 'orb-fallback', 'status': 'no_features'}
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]
        
        if len(matches) < 10:
            return None, {'method': 'orb-fallback', 'status': 'insufficient_matches'}
        
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None, {'method': 'orb-fallback', 'status': 'homography_failed'}
        
        h, w = img1.shape[:2]
        aligned = cv2.warpPerspective(img2, H, (w, h))
        
        inliers = int(np.sum(mask))
        confidence = (inliers / len(matches)) * 100 if len(matches) > 0 else 0
        
        return aligned, {
            'method': 'orb-fallback',
            'status': 'success',
            'matches': len(matches),
            'inliers': inliers,
            'confidence': confidence
        }
    
    # ==================== VISUALIZATION ====================
    
    def _create_overlay(self, image, mask):
        """Create cyan glare overlay"""
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 255]
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    def _create_alignment_viz(self, img1, img2, img2_aligned):
        """Side-by-side alignment comparison"""
        h, w = img1.shape[:2]
        if h > 400:
            scale = 400 / h
            img1_s = cv2.resize(img1, (int(w*scale), int(h*scale)))
            img2_s = cv2.resize(img2, (int(w*scale), int(h*scale)))
            img2_a_s = cv2.resize(img2_aligned, (int(w*scale), int(h*scale)))
        else:
            img1_s, img2_s, img2_a_s = img1, img2, img2_aligned
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1_s, 'Image 1 (Ref)', (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(img2_s, 'Image 2 (Original)', (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(img2_a_s, 'Image 2 (Aligned)', (10, 30), font, 0.7, (0, 255, 0), 2)
        
        return np.hstack([img1_s, img2_s, img2_a_s])
    
    def _create_detection_viz(self, img1, img2, mask1, mask2):
        """Detection comparison"""
        overlay1 = self._create_overlay(img1, mask1)
        overlay2 = self._create_overlay(img2, mask2)
        
        h, w = img1.shape[:2]
        if h > 400:
            scale = 400 / h
            overlay1 = cv2.resize(overlay1, (int(w*scale), int(h*scale)))
            overlay2 = cv2.resize(overlay2, (int(w*scale), int(h*scale)))
        
        return np.hstack([overlay1, overlay2])
    
    def _create_final_comparison(self, img1, img2, result):
        """Final before/after comparison"""
        h, w = img1.shape[:2]
        if h > 400:
            scale = 400 / h
            img1_s = cv2.resize(img1, (int(w*scale), int(h*scale)))
            img2_s = cv2.resize(img2, (int(w*scale), int(h*scale)))
            result_s = cv2.resize(result, (int(w*scale), int(h*scale)))
        else:
            img1_s, img2_s, result_s = img1, img2, result
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1_s, 'Input 1', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(img2_s, 'Input 2', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(result_s, 'RESULT (Glare-Free)', (10, 30), font, 1, (0, 255, 0), 2)
        
        return np.hstack([img1_s, img2_s, result_s])
    
    def _create_summary(self, img1_name, img2_name, align_info, 
                       stats1, stats2, recon_stats, eval_metrics, timestamp):
        """Create processing summary"""
        return {
            'timestamp': timestamp,
            'inputs': {'image1': img1_name, 'image2': img2_name},
            'alignment': align_info,
            'glare_detection': {'image1': stats1, 'image2': stats2},
            'reconstruction': recon_stats,
            'evaluation': eval_metrics
        }


def main():
    """Command-line interface"""
    if len(sys.argv) < 3:
        print("\n❌ Usage: python complete_glare_pipeline.py <img1> <img2> [options]")
        print("\nOptions:")
        print("  --tune          Interactive detection tuning")
        print("  --basic         Use basic detection")
        print("  --method METHOD Alignment: auto/orb/sift/ecc")
        print("  --resize WxH    Resize (e.g., 1920x1080)")
        sys.exit(1)
    
    img1_filename = sys.argv[1]
    img2_filename = sys.argv[2]
    
    tune = '--tune' in sys.argv
    use_basic = '--basic' in sys.argv
    
    method = 'auto'
    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]
    
    resize = None
    if '--resize' in sys.argv:
        idx = sys.argv.index('--resize')
        if idx + 1 < len(sys.argv):
            resize = sys.argv[idx + 1]
    
    pipeline = CompleteGlarePipeline()
    
    if use_basic:
        pipeline.use_enhanced_detection = False
    
    try:
        results = pipeline.process(
            img1_filename, img2_filename,
            tune_detection=tune,
            alignment_method=method,
            target_resolution=resize
        )
        
        if results['success']:
            print("\n✅ SUCCESS!")
            print(f"📊 Quality: {results['reconstruction']['recoverable_percentage']:.1f}% glare removed")
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()