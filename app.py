import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from image_stitcher import ImageStitcher
from utils import validate_image, preprocess_image, create_download_link

def main():
    st.set_page_config(
        page_title="Drone Image Stitcher",
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("🗺️ Drone Image Stitcher")
    st.markdown("Upload multiple drone images to create a cohesive map using advanced computer vision algorithms.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Feature detector selection
        detector_type = st.selectbox(
            "Feature Detector",
            ["SIFT", "ORB"],
            help="Choose the feature detection algorithm"
        )
        
        # Matching threshold
        match_threshold = st.slider(
            "Match Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Threshold for feature matching (lower = more matches)"
        )
        
        # RANSAC threshold
        ransac_threshold = st.slider(
            "RANSAC Threshold",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Threshold for RANSAC outlier detection"
        )
        
        # Output format
        output_format = st.selectbox(
            "Output Format",
            ["PNG", "JPEG"],
            help="Format for the stitched image"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Images")
        uploaded_files = st.file_uploader(
            "Choose drone images",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            help="Upload 2 or more drone images to stitch together"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} images")
            
            # Display thumbnails
            if st.checkbox("Show uploaded images"):
                cols = st.columns(min(3, len(uploaded_files)))
                for i, uploaded_file in enumerate(uploaded_files[:6]):  # Show max 6 thumbnails
                    with cols[i % 3]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=uploaded_file.name, use_column_width=True)
    
    with col2:
        st.header("🔧 Processing")
        
        if uploaded_files and len(uploaded_files) >= 2:
            if st.button("🚀 Start Stitching", type="primary"):
                try:
                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load and validate images
                    status_text.text("Loading and validating images...")
                    progress_bar.progress(10)
                    
                    images = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        if not validate_image(image_bytes):
                            st.error(f"Invalid image: {uploaded_file.name}")
                            return
                        
                        # Convert to OpenCV format
                        image = np.array(Image.open(io.BytesIO(image_bytes)))
                        if len(image.shape) == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        # Preprocess image
                        image = preprocess_image(image)
                        images.append(image)
                        
                        progress = 10 + (i + 1) * 20 // len(uploaded_files)
                        progress_bar.progress(progress)
                    
                    # Initialize stitcher
                    status_text.text("Initializing image stitcher...")
                    progress_bar.progress(35)
                    
                    stitcher = ImageStitcher(
                        detector_type=detector_type,
                        match_threshold=match_threshold,
                        ransac_threshold=ransac_threshold
                    )
                    
                    # Perform stitching
                    status_text.text("Detecting features and matching...")
                    progress_bar.progress(50)
                    
                    result = stitcher.stitch_images(images, progress_callback=lambda p: progress_bar.progress(50 + p * 45 // 100))
                    
                    if result is not None:
                        status_text.text("Stitching completed successfully!")
                        progress_bar.progress(100)
                        
                        # Convert back to RGB for display
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        result_pil = Image.fromarray(result_rgb)
                        
                        # Display result
                        st.success("✅ Stitching completed successfully!")
                        
                        # Show stitched image
                        st.subheader("📍 Stitched Map")
                        st.image(result_pil, caption="Stitched Drone Map", use_column_width=True)
                        
                        # Create download link
                        download_link = create_download_link(result_pil, output_format)
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Display image info
                        st.info(f"Result dimensions: {result.shape[1]} x {result.shape[0]} pixels")
                        
                    else:
                        st.error("❌ Stitching failed. Please try with different images or adjust settings.")
                        st.info("Tips:\n- Ensure images have sufficient overlap\n- Try adjusting the match threshold\n- Use images from the same height/angle")
                        
                except Exception as e:
                    st.error(f"An error occurred during stitching: {str(e)}")
                    st.info("Please check your images and try again with different settings.")
        else:
            st.info("Please upload at least 2 images to start stitching.")
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### Image Stitching Process
        
        1. **Feature Detection**: The algorithm detects key features in each image using SIFT or ORB detectors
        2. **Feature Matching**: Features are matched between overlapping images
        3. **Homography Estimation**: The geometric transformation between images is calculated
        4. **Image Warping**: Images are warped and aligned based on the homography
        5. **Blending**: Overlapping regions are blended to create a seamless result
        
        ### Tips for Best Results
        - Ensure images have 20-50% overlap
        - Use images taken from similar heights and angles
        - Avoid images with too much motion blur
        - Sequential images work better than random collections
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and OpenCV • Drone Image Stitching Tool")

if __name__ == "__main__":
    main()
