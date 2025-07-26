import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional
import io

class NDVIAnalyzer:
    """
    A class for computing NDVI (Normalized Difference Vegetation Index) from drone images.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    
    def __init__(self):
        """Initialize the NDVI analyzer."""
        self.ndvi_colormap = cm.get_cmap('RdYlGn')  # Red-Yellow-Green colormap
        
    def extract_bands(self, image: np.ndarray, band_type: str = "standard_rgb") -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Red and NIR bands from the image.
        
        Args:
            image: Input image (BGR format)
            band_type: Type of band extraction ("standard_rgb", "modified_rgb", "infrared_rgb")
            
        Returns:
            Tuple of (red_band, nir_band)
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must have 3 channels")
            
        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if band_type == "standard_rgb":
            # Standard RGB: Use red channel as red, assume green as pseudo-NIR
            red_band = rgb_image[:, :, 0].astype(np.float32)
            nir_band = rgb_image[:, :, 1].astype(np.float32)  # Green as pseudo-NIR
            
        elif band_type == "modified_rgb":
            # Modified RGB: Red channel as red, blue channel as pseudo-NIR
            red_band = rgb_image[:, :, 0].astype(np.float32)
            nir_band = rgb_image[:, :, 2].astype(np.float32)  # Blue as pseudo-NIR
            
        elif band_type == "infrared_rgb":
            # Infrared RGB: Assume NIR is in red channel, visible red in green
            nir_band = rgb_image[:, :, 0].astype(np.float32)  # NIR in red channel
            red_band = rgb_image[:, :, 1].astype(np.float32)  # Red in green channel
            
        else:
            raise ValueError(f"Unsupported band type: {band_type}")
            
        return red_band, nir_band
    
    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI from red and NIR bands.
        
        Args:
            red_band: Red band array
            nir_band: NIR band array
            
        Returns:
            NDVI array with values between -1 and 1
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-7
        
        # Calculate NDVI
        numerator = nir_band - red_band
        denominator = nir_band + red_band + epsilon
        
        ndvi = numerator / denominator
        
        # Clip values to valid NDVI range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def create_ndvi_visualization(self, ndvi: np.ndarray, colormap: str = "RdYlGn") -> np.ndarray:
        """
        Create a colored visualization of NDVI values.
        
        Args:
            ndvi: NDVI array
            colormap: Matplotlib colormap name
            
        Returns:
            RGB visualization of NDVI
        """
        # Normalize NDVI from [-1, 1] to [0, 1] for colormap
        ndvi_normalized = (ndvi + 1) / 2
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored_ndvi = cmap(ndvi_normalized)
        
        # Convert to 8-bit RGB
        rgb_ndvi = (colored_ndvi[:, :, :3] * 255).astype(np.uint8)
        
        return rgb_ndvi
    
    def analyze_vegetation_health(self, ndvi: np.ndarray) -> dict:
        """
        Analyze vegetation health from NDVI values.
        
        Args:
            ndvi: NDVI array
            
        Returns:
            Dictionary with vegetation analysis
        """
        # Define NDVI thresholds
        water_mask = ndvi < -0.3
        bare_soil_mask = (ndvi >= -0.3) & (ndvi < 0.1)
        sparse_vegetation_mask = (ndvi >= 0.1) & (ndvi < 0.3)
        moderate_vegetation_mask = (ndvi >= 0.3) & (ndvi < 0.6)
        dense_vegetation_mask = ndvi >= 0.6
        
        total_pixels = ndvi.size
        
        analysis = {
            "mean_ndvi": float(np.mean(ndvi)),
            "std_ndvi": float(np.std(ndvi)),
            "min_ndvi": float(np.min(ndvi)),
            "max_ndvi": float(np.max(ndvi)),
            "water_percentage": float(np.sum(water_mask) / total_pixels * 100),
            "bare_soil_percentage": float(np.sum(bare_soil_mask) / total_pixels * 100),
            "sparse_vegetation_percentage": float(np.sum(sparse_vegetation_mask) / total_pixels * 100),
            "moderate_vegetation_percentage": float(np.sum(moderate_vegetation_mask) / total_pixels * 100),
            "dense_vegetation_percentage": float(np.sum(dense_vegetation_mask) / total_pixels * 100),
            "vegetation_coverage": float(np.sum(ndvi > 0.1) / total_pixels * 100)  # Above 0.1 is considered vegetation
        }
        
        return analysis
    
    def create_vegetation_mask(self, ndvi: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """
        Create a binary mask for vegetation areas.
        
        Args:
            ndvi: NDVI array
            threshold: NDVI threshold for vegetation detection
            
        Returns:
            Binary mask where vegetation areas are True
        """
        return ndvi > threshold
    
    def process_image(self, image: np.ndarray, band_type: str = "standard_rgb") -> dict:
        """
        Complete NDVI processing pipeline for a single image.
        
        Args:
            image: Input image (BGR format)
            band_type: Type of band extraction
            
        Returns:
            Dictionary with all NDVI analysis results
        """
        try:
            # Extract bands
            red_band, nir_band = self.extract_bands(image, band_type)
            
            # Calculate NDVI
            ndvi = self.calculate_ndvi(red_band, nir_band)
            
            # Create visualization
            ndvi_visualization = self.create_ndvi_visualization(ndvi)
            
            # Analyze vegetation health
            vegetation_analysis = self.analyze_vegetation_health(ndvi)
            
            # Create vegetation mask
            vegetation_mask = self.create_vegetation_mask(ndvi)
            
            return {
                "ndvi": ndvi,
                "ndvi_visualization": ndvi_visualization,
                "vegetation_analysis": vegetation_analysis,
                "vegetation_mask": vegetation_mask,
                "red_band": red_band,
                "nir_band": nir_band
            }
            
        except Exception as e:
            raise Exception(f"NDVI processing failed: {str(e)}")
    
    def create_comparison_plot(self, original_image: np.ndarray, ndvi_result: dict) -> bytes:
        """
        Create a comparison plot showing original image, NDVI, and analysis.
        
        Args:
            original_image: Original BGR image
            ndvi_result: Result from process_image
            
        Returns:
            PNG image bytes of the comparison plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('NDVI Analysis Results', fontsize=16, fontweight='bold')
        
        # Original image (convert BGR to RGB for display)
        rgb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(rgb_original)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # NDVI visualization
        axes[0, 1].imshow(ndvi_result["ndvi_visualization"])
        axes[0, 1].set_title('NDVI Visualization')
        axes[0, 1].axis('off')
        
        # NDVI histogram
        axes[1, 0].hist(ndvi_result["ndvi"].flatten(), bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('NDVI Distribution')
        axes[1, 0].set_xlabel('NDVI Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Vegetation analysis text
        analysis = ndvi_result["vegetation_analysis"]
        analysis_text = f"""
Vegetation Analysis:
• Mean NDVI: {analysis['mean_ndvi']:.3f}
• Vegetation Coverage: {analysis['vegetation_coverage']:.1f}%
• Dense Vegetation: {analysis['dense_vegetation_percentage']:.1f}%
• Moderate Vegetation: {analysis['moderate_vegetation_percentage']:.1f}%
• Sparse Vegetation: {analysis['sparse_vegetation_percentage']:.1f}%
• Bare Soil: {analysis['bare_soil_percentage']:.1f}%
• Water Bodies: {analysis['water_percentage']:.1f}%
        """
        
        axes[1, 1].text(0.05, 0.95, analysis_text.strip(), transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
    
    def get_ndvi_legend(self) -> bytes:
        """
        Create an NDVI color scale legend.
        
        Returns:
            PNG image bytes of the legend
        """
        fig, ax = plt.subplots(figsize=(8, 2))
        
        # Create gradient
        gradient = np.linspace(-1, 1, 256).reshape(1, -1)
        
        # Display gradient with colormap
        im = ax.imshow(gradient, aspect='auto', cmap='RdYlGn')
        
        # Set labels
        ax.set_xlim(0, 255)
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
        ax.set_yticks([])
        ax.set_xlabel('NDVI Value')
        ax.set_title('NDVI Color Scale (Red: No Vegetation → Green: Dense Vegetation)')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()