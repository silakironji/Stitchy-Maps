import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional

def validate_image(image_bytes: bytes) -> bool:
    """
    Validate if the uploaded file is a valid image.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        # Try to open with PIL
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False

def preprocess_image(image: np.ndarray, max_dimension: int = 1500) -> np.ndarray:
    """
    Preprocess image by resizing if too large and ensuring proper format.
    
    Args:
        image: Input image
        max_dimension: Maximum dimension for resizing
        
    Returns:
        Preprocessed image
    """
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Resize if image is too large
    if max(height, width) > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * max_dimension / width)
        else:
            new_height = max_dimension
            new_width = int(width * max_dimension / height)
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Ensure image is in BGR format for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB, then to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Assume RGB, convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

def create_download_link(image: Image.Image, format_type: str = "PNG") -> str:
    """
    Create a download link for the stitched image.
    
    Args:
        image: PIL Image object
        format_type: Output format (PNG or JPEG)
        
    Returns:
        HTML download link
    """
    # Convert image to bytes
    img_buffer = io.BytesIO()
    
    if format_type.upper() == "JPEG":
        # Convert to RGB if saving as JPEG (no transparency)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(img_buffer, format="JPEG", quality=95)
        mime_type = "image/jpeg"
        file_extension = "jpg"
    else:  # PNG
        image.save(img_buffer, format="PNG")
        mime_type = "image/png"
        file_extension = "png"
    
    img_buffer.seek(0)
    
    # Encode to base64
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    # Create download link
    filename = f"stitched_map.{file_extension}"
    download_link = f"""
    <a href="data:{mime_type};base64,{img_b64}" download="{filename}">
        <button style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        ">
            📥 Download Stitched Map ({format_type})
        </button>
    </a>
    """
    
    return download_link

def calculate_image_stats(image: np.ndarray) -> dict:
    """
    Calculate basic statistics for an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image statistics
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    stats = {
        "width": width,
        "height": height,
        "channels": channels,
        "total_pixels": width * height,
        "aspect_ratio": round(width / height, 2)
    }
    
    if channels == 3:
        # Calculate mean values for each channel
        stats["mean_bgr"] = [float(np.mean(image[:, :, i])) for i in range(3)]
    else:
        stats["mean_intensity"] = float(np.mean(image))
    
    return stats

def enhance_image_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Clipping limit for CLAHE
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def detect_image_orientation(image: np.ndarray) -> float:
    """
    Detect the orientation of an image using edge detection.
    
    Args:
        image: Input image
        
    Returns:
        Estimated rotation angle in degrees
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        # Calculate the most common angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # Convert to rotation angle
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
        
        # Return the median angle
        return float(np.median(angles))
    
    return 0.0
