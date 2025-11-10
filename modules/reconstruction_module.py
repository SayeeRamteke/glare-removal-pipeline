"""
Module 4: Image Reconstruction
Purpose: Combine information from both images to create glare-free result

INPUT:
    - img1: numpy array (H, W, 3) - First image with glare in certain regions
    - img2_aligned: numpy array (H, W, 3) - Second image (aligned) with glare in different regions
    - mask1: numpy array (H, W) - Binary mask (255=glare, 0=no glare) for img1
    - mask2: numpy array (H, W) - Binary mask (255=glare, 0=no glare) for img2

OUTPUT:
    - result: numpy array (H, W, 3) - Reconstructed glare-free image
    - overlap_mask: numpy array (H, W) - Regions where both images had glare
    - stats: dict - Statistics about the reconstruction process
"""

import cv2
import numpy as np
from typing import Tuple, Dict


class ReconstructionModule:
    def __init__(self, blend_method='blended'):
        """
        Initialize reconstruction module.
        
        Args:
            blend_method (str): 'simple' or 'blended'
                - 'simple': Direct pixel replacement
                - 'blended': Smooth transition at boundaries
        """
        self.blend_method = blend_method
        self.overlap_threshold = 50.0  # percentage
        
    def reconstruct(self, img1: np.ndarray, img2_aligned: np.ndarray, 
                   mask1: np.ndarray, mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Main reconstruction function.
        
        Args:
            img1: First image (BGR format)
            img2_aligned: Second image aligned to first (BGR format)
            mask1: Glare mask for first image (binary: 0 or 255)
            mask2: Glare mask for second image (binary: 0 or 255)
            
        Returns:
            result: Reconstructed image
            overlap_mask: Mask showing overlapping glare regions
            stats: Dictionary with reconstruction statistics
        """
        # Validate inputs
        self._validate_inputs(img1, img2_aligned, mask1, mask2)
        
        # Handle overlapping glare
        overlap_mask = self._handle_overlapping_glare(mask1, mask2)
        
        # Perform reconstruction based on selected method
        if self.blend_method == 'simple':
            result = self._reconstruct_simple(img1, img2_aligned, mask1, mask2, overlap_mask)
        elif self.blend_method == 'hybrid':  # NEW!
            result = self._reconstruct_hybrid(img1, img2_aligned, mask1, mask2, overlap_mask)
        else:  # blended
            result = self._reconstruct_blended(img1, img2_aligned, mask1, mask2, overlap_mask)
        
        # Compute statistics
        stats = self._compute_statistics(mask1, mask2, overlap_mask)
        
        return result, overlap_mask, stats
    
    def _validate_inputs(self, img1: np.ndarray, img2_aligned: np.ndarray, 
                        mask1: np.ndarray, mask2: np.ndarray) -> None:
        """Validate input dimensions and types."""
        if img1.shape != img2_aligned.shape:
            raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2_aligned.shape}")
        
        if mask1.shape[:2] != img1.shape[:2]:
            raise ValueError(f"Mask1 shape {mask1.shape} doesn't match image shape {img1.shape[:2]}")
        
        if mask2.shape[:2] != img1.shape[:2]:
            raise ValueError(f"Mask2 shape {mask2.shape} doesn't match image shape {img1.shape[:2]}")
        
        if len(img1.shape) != 3 or img1.shape[2] != 3:
            raise ValueError(f"Images must be 3-channel BGR, got shape {img1.shape}")
    
    def _handle_overlapping_glare(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Find and report overlapping glare regions.
        
        Args:
            mask1: Binary mask for image 1
            mask2: Binary mask for image 2
            
        Returns:
            overlap_mask: Binary mask showing overlapping regions
        """
        # Find pixels where BOTH images have glare
        overlap = cv2.bitwise_and(mask1, mask2)
        
        # Calculate overlap statistics
        total_glare_pixels = np.sum(mask1 > 0)
        overlap_pixels = np.sum(overlap > 0)
        
        if total_glare_pixels > 0:
            overlap_percentage = (overlap_pixels / total_glare_pixels) * 100
            
            if overlap_percentage > self.overlap_threshold:
                print(f"\n⚠️  WARNING: {overlap_percentage:.1f}% of glare overlaps!")
                print("    Reconstruction quality may be reduced in these regions.")
                print("    Consider taking a third image from a different angle.\n")
        
        return overlap
    
    def _reconstruct_simple(self, img1: np.ndarray, img2_aligned: np.ndarray,
                           mask1: np.ndarray, mask2: np.ndarray, 
                           overlap_mask: np.ndarray) -> np.ndarray:
        """
        Simple pixel-wise reconstruction.
        Strategy:
        - Start with img1 as base
        - Where img1 has glare, use img2
        - In overlap regions, average both images
        
        Args:
            img1: First image
            img2_aligned: Aligned second image
            mask1: Glare mask for img1
            mask2: Glare mask for img2
            overlap_mask: Overlapping glare regions
            
        Returns:
            result: Reconstructed image
        """
        # Start with image1 as base
        result = img1.copy()
        
        # Create boolean masks for easier indexing
        glare1 = mask1 > 0
        glare2 = mask2 > 0
        overlap = overlap_mask > 0
        
        # Where only img1 has glare (and img2 doesn't), use img2
        use_img2 = glare1 & ~glare2
        result[use_img2] = img2_aligned[use_img2]
        
        # In overlap regions, average both images (best we can do)
        if np.any(overlap):
            result[overlap] = cv2.addWeighted(
                img1[overlap], 0.5, 
                img2_aligned[overlap], 0.5, 0
            )
        
        return result
    
    def _reconstruct_blended(self, img1: np.ndarray, img2_aligned: np.ndarray,
                            mask1: np.ndarray, mask2: np.ndarray,
                            overlap_mask: np.ndarray) -> np.ndarray:
        """
        Blended reconstruction with smooth transitions.
        Uses distance transform to create smooth weight transitions at glare boundaries.
        
        Args:
            img1: First image
            img2_aligned: Aligned second image
            mask1: Glare mask for img1
            mask2: Glare mask for img2
            overlap_mask: Overlapping glare regions
            
        Returns:
            result: Reconstructed image with smooth blending
        """
        result = img1.copy().astype(np.float32)
        img2_float = img2_aligned.astype(np.float32)
        
        # Create blending weights based on distance from glare
        weights = self._create_blend_weights(mask1)
        
        # Convert masks to boolean for indexing
        glare1 = mask1 > 0
        overlap = overlap_mask > 0
        
        # Expand weights to 3 channels (BGR)
        weights_3ch = np.stack([weights, weights, weights], axis=2)
        
        # Blend in glare regions of img1 (excluding overlap)
        glare_only = glare1 & ~overlap
        if np.any(glare_only):
            result[glare_only] = (
                weights_3ch[glare_only] * result[glare_only] + 
                (1 - weights_3ch[glare_only]) * img2_float[glare_only]
            )
        
        # Handle overlap regions (average both images)
        if np.any(overlap):
            result[overlap] = 0.5 * result[overlap] + 0.5 * img2_float[overlap]
        
        # Add smooth transition zone around glare boundaries
        result = self._apply_boundary_smoothing(result, img1, img2_float, mask1)
        
        return result.astype(np.uint8)
    
    def _create_blend_weights(self, mask: np.ndarray, 
                             transition_width: int = 10) -> np.ndarray:
        """
        Create smooth blending weights using distance transform.
        Center of glare: weight = 0 (use img2 fully)
        Edge of glare: weight transitions from 0 to 1
        Outside glare: weight = 1 (use img1 fully)
        
        Args:
            mask: Binary glare mask
            transition_width: Width of transition zone in pixels
            
        Returns:
            weights: Normalized weights (0 to 1)
        """
        # Distance transform gives distance to nearest glare pixel
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        
        # Normalize distance to create smooth transition
        weights = np.clip(dist / transition_width, 0, 1)
        
        return weights
    
    def _apply_boundary_smoothing(self, result: np.ndarray, img1: np.ndarray,
                                  img2_float: np.ndarray, mask1: np.ndarray,
                                  smoothing_radius: int = 5) -> np.ndarray:
        """
        Apply additional smoothing at glare boundaries to reduce visible seams.
        
        Args:
            result: Current reconstruction result
            img1: Original first image
            img2_float: Second image (float)
            mask1: Glare mask
            smoothing_radius: Radius for boundary smoothing
            
        Returns:
            Smoothed result
        """
        # Find boundary of glare regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(mask1, kernel, iterations=1)
        eroded = cv2.erode(mask1, kernel, iterations=1)
        boundary = dilated - eroded
        
        # Apply Gaussian blur in boundary regions for smooth transition
        if np.any(boundary > 0):
            blurred = cv2.GaussianBlur(result, (smoothing_radius*2+1, smoothing_radius*2+1), 0)
            result[boundary > 0] = blurred[boundary > 0]
        
        return result
    def _reconstruct_hybrid(self, img1, img2_aligned, mask1, mask2, overlap_mask):
        """
        Hybrid approach: Use img2 for content, then inpaint boundaries
        """
        # Step 1: Basic reconstruction from img2
        result = img1.copy()
        glare1 = mask1 > 0
        glare2 = mask2 > 0
        overlap = overlap_mask > 0
    
        # Use img2 where img1 has glare
        use_img2 = glare1 & ~glare2
        result[use_img2] = img2_aligned[use_img2]
    
        # Step 2: Color harmonization (match img2 to img1's color tone)
        result = self._harmonize_colors(result, img1, mask1)
    
        # Step 3: Inpaint the boundaries for seamless transition
        boundary_mask = self._create_boundary_mask(mask1, width=15)
        result = cv2.inpaint(result, boundary_mask, 3, cv2.INPAINT_TELEA)
    
        # Step 4: Handle overlap regions
        if np.any(overlap):
        # For overlap, inpaint using surrounding context
            result = cv2.inpaint(result, overlap.astype(np.uint8) * 255, 
                            5, cv2.INPAINT_NS)
    
        return result

    def _harmonize_colors(self, result, reference, mask):
        """Match color statistics between reconstructed and original regions"""
        # Convert to LAB color space for better color matching
        result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    
        # Create a border region around glare for sampling
        kernel = np.ones((15, 15), np.uint8)
        border_region = cv2.dilate(mask, kernel) - mask
    
        if np.sum(border_region) > 0:
        # Match mean and std of L, A, B channels
            for i in range(3):
                src_mean = result_lab[:,:,i][border_region > 0].mean()
                src_std = result_lab[:,:,i][border_region > 0].std()
                dst_mean = ref_lab[:,:,i][border_region > 0].mean()
                dst_std = ref_lab[:,:,i][border_region > 0].std()
            
            # Apply color transfer in glare region
            glare_region = mask > 0
            result_lab[:,:,i][glare_region] = (
                (result_lab[:,:,i][glare_region] - src_mean) * 
                (dst_std / src_std) + dst_mean
            )
    
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def _create_boundary_mask(self, mask, width=15):
        """Create a mask of the boundary region"""
        kernel = np.ones((width, width), np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        boundary = dilated - eroded
        return boundary.astype(np.uint8)
    
    def _compute_statistics(self, mask1: np.ndarray, mask2: np.ndarray,
                           overlap_mask: np.ndarray) -> Dict:
        """
        Compute reconstruction statistics.
        
        Returns:
            Dictionary containing:
                - glare1_pixels: Number of glare pixels in img1
                - glare2_pixels: Number of glare pixels in img2
                - overlap_pixels: Number of overlapping glare pixels
                - overlap_percentage: Percentage of overlap relative to total glare
                - recoverable_percentage: Percentage of glare that can be fully recovered
        """
        glare1_pixels = int(np.sum(mask1 > 0))
        glare2_pixels = int(np.sum(mask2 > 0))
        overlap_pixels = int(np.sum(overlap_mask > 0))
        
        total_glare = glare1_pixels
        recoverable_pixels = glare1_pixels - overlap_pixels
        
        overlap_pct = (overlap_pixels / total_glare * 100) if total_glare > 0 else 0
        recoverable_pct = (recoverable_pixels / total_glare * 100) if total_glare > 0 else 100
        
        return {
            'glare1_pixels': glare1_pixels,
            'glare2_pixels': glare2_pixels,
            'overlap_pixels': overlap_pixels,
            'overlap_percentage': round(overlap_pct, 2),
            'recoverable_percentage': round(recoverable_pct, 2),
            'method_used': self.blend_method
        }
    
    def visualize_reconstruction(self, img1: np.ndarray, img2_aligned: np.ndarray,
                                result: np.ndarray, mask1: np.ndarray, 
                                mask2: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
        """
        Create visualization comparing input images and result.
        
        Returns:
            Composite image showing comparison
        """
        # Resize for display if too large
        h, w = img1.shape[:2]
        if h > 600:
            scale = 600 / h
            new_h, new_w = int(h * scale), int(w * scale)
            img1_vis = cv2.resize(img1, (new_w, new_h))
            img2_vis = cv2.resize(img2_aligned, (new_w, new_h))
            result_vis = cv2.resize(result, (new_w, new_h))
            mask1_vis = cv2.resize(mask1, (new_w, new_h))
            overlap_vis = cv2.resize(overlap_mask, (new_w, new_h))
        else:
            img1_vis = img1.copy()
            img2_vis = img2_aligned.copy()
            result_vis = result.copy()
            mask1_vis = mask1.copy()
            overlap_vis = overlap_mask.copy()
        
        # Convert masks to color for visualization
        mask1_colored = cv2.applyColorMap(mask1_vis, cv2.COLORMAP_JET)
        overlap_colored = cv2.applyColorMap(overlap_vis, cv2.COLORMAP_HOT)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1_vis, 'Image 1', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(img2_vis, 'Image 2 (aligned)', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(result_vis, 'Result', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(mask1_colored, 'Glare Mask 1', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(overlap_colored, 'Overlap', (10, 30), font, 1, (255, 255, 255), 2)
        
        # Stack images
        row1 = np.hstack([img1_vis, img2_vis, result_vis])
        row2 = np.hstack([mask1_colored, overlap_colored, 
                         np.zeros_like(overlap_colored)])
        
        composite = np.vstack([row1, row2])
        
        return composite