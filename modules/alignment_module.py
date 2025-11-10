import cv2
import numpy as np

def order_points(pts):
    """Order contour points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """Warp image to a top-down view using 4 corner points"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def detect_document(image):
    """Detect document using edge detection and contours"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 30, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    image_area = image.shape[0] * image.shape[1]
    valid_contours = [
        c for c in cnts if image_area * 0.05 < cv2.contourArea(c) < image_area * 0.6
    ]
    if not valid_contours:
        return None

    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    for c in valid_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    largest = max(valid_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return box.astype(int)


def process_documents(img1, img2, output_width=None):
    """
    Pipeline: ORB Alignment → Edge Detection Cropping
    
    1. Align Image 2 to Image 1 using ORB feature matching
    2. Detect document edges in aligned Image 1
    3. Crop both aligned images using the same perspective transform
    
    Args:
        img1: First image (reference)
        img2: Second image (will be aligned to img1)
        output_width: Final output width (optional), height will be scaled proportionally.
    
    Returns:
        (cropped_img1, cropped_img2)
    """
    
    # STEP 1: ORB Alignment
    print("\n[1/3] Aligning images using ORB feature matching...")
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    print(f"      Keypoints detected - Image 1: {len(kp1)}, Image 2: {len(kp2)}")
    
    if des1 is None or des2 is None:
        print("       No descriptors found, skipping alignment")
        aligned_img1 = img1
        aligned_img2 = img2
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        print(f"      Good matches found: {len(good_matches)}")
        
        if len(good_matches) < 10:
            print("        Not enough matches, skipping alignment")
            aligned_img1 = img1
            aligned_img2 = img2
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print("         Homography failed, skipping alignment")
                aligned_img1 = img1
                aligned_img2 = img2
            else:
                h, w = img1.shape[:2]
                aligned_img2 = cv2.warpPerspective(img2, H, (w, h))
                aligned_img1 = img1
                print("      ✓ Image 2 successfully aligned to Image 1")
    
    # STEP 2: Document Detection
    print("\n[2/3] Detecting document edges...")
    
    pts = detect_document(aligned_img1)
    
    if pts is None:
        print("         Document not detected, returning resized images")
        if output_width:
            aspect_ratio = img1.shape[0] / img1.shape[1]
            output_height = int(output_width * aspect_ratio)
            result1 = cv2.resize(aligned_img1, (output_width, output_height))
            result2 = cv2.resize(aligned_img2, (output_width, output_height))
        else:
            result1 = aligned_img1
            result2 = aligned_img2
        return result1, result2
    
    print("        Document edges detected")
    
    # STEP 3: Crop Both Images
    print("\n[3/3] Cropping and resizing both images...")
    
    cropped1 = four_point_transform(aligned_img1, pts)
    cropped2 = four_point_transform(aligned_img2, pts)
    
    # Dynamically calculate the output size if not provided
    if output_width:
        aspect_ratio = cropped1.shape[0] / cropped1.shape[1]
        output_height = int(output_width * aspect_ratio)
        result1 = cv2.resize(cropped1, (output_width, output_height))
        result2 = cv2.resize(cropped2, (output_width, output_height))
    else:
        result1 = cropped1
        result2 = cropped2
    
    print(f"      ✓ Final output size: ({result1.shape[1]}, {result1.shape[0]})")
    
    return result1, result2


if __name__ == "__main__":
    # Load images
    img1_path = "doc1.jpg"
    img2_path = "doc2.jpg"
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images. Check file paths.")
    
    print(f"\nInput - Image 1: {img1.shape}")
    print(f"Input - Image 2: {img2.shape}\n")
    
    # Process through pipeline with output width of 800 pixels
    result1, result2 = process_documents(img1, img2)
    
    # Save results
    cv2.imwrite("result1_aligned_cropped.jpg", result1)
    cv2.imwrite("result2_aligned_cropped.jpg", result2)
    
    print("\n✓ Processing complete!")
    print(f"Output - Image 1: {result1.shape}")
    print(f"Output - Image 2: {result2.shape}")
    print("\nSaved: result1_aligned_cropped.jpg")
    print("Saved: result2_aligned_cropped.jpg")