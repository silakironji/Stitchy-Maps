import cv2
import numpy as np
from typing import List, Optional, Callable

class ImageStitcher:
    """
    A class for stitching multiple drone images together using OpenCV.
    Supports SIFT and ORB feature detectors with configurable parameters.
    """
    
    def __init__(self, detector_type: str = "SIFT", match_threshold: float = 0.75, ransac_threshold: float = 5.0):
        """
        Initialize the ImageStitcher.
        
        Args:
            detector_type: Type of feature detector ("SIFT" or "ORB")
            match_threshold: Threshold for feature matching
            ransac_threshold: Threshold for RANSAC outlier detection
        """
        self.detector_type = detector_type
        self.match_threshold = match_threshold
        self.ransac_threshold = ransac_threshold
        
        # Initialize feature detector
        if detector_type == "SIFT":
            self.detector = cv2.SIFT_create()
        elif detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=5000)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
        
        # Initialize matcher
        if detector_type == "SIFT":
            self.matcher = cv2.BFMatcher()
        else:  # ORB
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def detect_and_describe(self, image: np.ndarray) -> tuple:
        """
        Detect keypoints and compute descriptors for an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match features between two sets of descriptors.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        if self.detector_type == "SIFT":
            # Use KNN matching for SIFT
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_threshold * n.distance:
                        good_matches.append(m)
        else:  # ORB
            # Use simple matching for ORB
            matches = self.matcher.match(desc1, desc2)
            # Sort by distance and take the best matches
            matches = sorted(matches, key=lambda x: x.distance)
            # Take top matches based on threshold
            num_good_matches = int(len(matches) * self.match_threshold)
            good_matches = matches[:num_good_matches]
        
        return good_matches
    
    def find_homography(self, kp1: List, kp2: List, matches: List) -> Optional[np.ndarray]:
        """
        Find homography matrix between matched keypoints.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches
            
        Returns:
            Homography matrix or None if insufficient matches
        """
        if len(matches) < 4:
            return None
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            self.ransac_threshold
        )
        
        return homography
    
    def warp_and_blend(self, img1: np.ndarray, img2: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """
        Warp and blend two images using the homography matrix.
        
        Args:
            img1: First image (to be warped)
            img2: Second image (reference)
            homography: Homography matrix
            
        Returns:
            Blended result image
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of the first image
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        
        # Transform corners using homography
        transformed_corners = cv2.perspectiveTransform(corners1, homography)
        
        # Combine corners to find the output size
        all_corners = np.concatenate([
            transformed_corners,
            np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        ])
        
        # Find bounding rectangle
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
        
        # Translation to ensure positive coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        
        # Update homography with translation
        homography_translated = translation.dot(homography)
        
        # Output size
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        # Warp first image
        warped_img1 = cv2.warpPerspective(img1, homography_translated, (output_width, output_height))
        
        # Create output image and place second image
        result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Place second image at the correct position
        result[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2
        
        # Create masks for blending
        mask1 = np.zeros((output_height, output_width), dtype=np.uint8)
        mask2 = np.zeros((output_height, output_width), dtype=np.uint8)
        
        # Mask for warped image
        mask1[warped_img1.sum(axis=2) > 0] = 255
        
        # Mask for second image
        mask2[-y_min:-y_min + h2, -x_min:-x_min + w2] = 255
        
        # Find overlap region
        overlap = cv2.bitwise_and(mask1, mask2)
        
        # Blend in overlap region
        if np.any(overlap):
            # Simple alpha blending in overlap region
            alpha = 0.5
            overlap_indices = overlap > 0
            
            result[overlap_indices] = (
                alpha * warped_img1[overlap_indices] + 
                (1 - alpha) * result[overlap_indices]
            ).astype(np.uint8)
        
        # Fill non-overlap regions
        non_overlap_warped = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
        result[non_overlap_warped > 0] = warped_img1[non_overlap_warped > 0]
        
        return result
    
    def stitch_two_images(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Stitch two images together.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Stitched image or None if stitching failed
        """
        # Detect features
        kp1, desc1 = self.detect_and_describe(img1)
        kp2, desc2 = self.detect_and_describe(img2)
        
        if desc1 is None or desc2 is None:
            return None
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 4:
            return None
        
        # Find homography
        homography = self.find_homography(kp1, kp2, matches)
        
        if homography is None:
            return None
        
        # Warp and blend
        result = self.warp_and_blend(img1, img2, homography)
        
        return result
    
    def stitch_images(self, images: List[np.ndarray], progress_callback: Optional[Callable] = None) -> Optional[np.ndarray]:
        """
        Stitch multiple images together.
        
        Args:
            images: List of images to stitch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Stitched result or None if stitching failed
        """
        if len(images) < 2:
            return None
        
        # Start with the first image
        result = images[0].copy()
        
        # Progressively stitch remaining images
        for i, img in enumerate(images[1:], 1):
            if progress_callback:
                progress = int((i / (len(images) - 1)) * 100)
                progress_callback(progress)
            
            result = self.stitch_two_images(result, img)
            
            if result is None:
                return None
        
        return result
