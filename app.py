import streamlit as st
try:
    import cv2
except ImportError:
    st.error("OpenCV is not installed. Please install opencv-python-headless.")
    st.stop()

import numpy as np
from PIL import Image
import io
import zipfile
from image_stitcher import ImageStitcher
from utils import validate_image, preprocess_image, create_download_link
from ndvi_analyzer import NDVIAnalyzer

def main():
    st.set_page_config(
        page_title="Drone Image Stitcher",
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("🗺️ Drone Image Stitcher & NDVI Analyzer")
    st.markdown("Upload multiple drone images to create cohesive maps and analyze vegetation health using advanced computer vision algorithms.")
    
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
        
        st.header("NDVI Analysis")
        
        # Enable NDVI analysis
        enable_ndvi = st.checkbox(
            "Enable NDVI Analysis",
            help="Analyze vegetation health using NDVI calculations"
        )
        
        if enable_ndvi:
            # Band type selection
            band_type = st.selectbox(
                "Band Configuration",
                ["standard_rgb", "modified_rgb", "infrared_rgb"],
                help="Select the band configuration for NDVI calculation"
            )
            
            # NDVI colormap
            ndvi_colormap = st.selectbox(
                "NDVI Colormap",
                ["RdYlGn", "viridis", "plasma", "coolwarm"],
                help="Color scheme for NDVI visualization"
            )
    
    # Main content area - use tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📁 Upload & Stitch", "🌱 NDVI Analysis", "ℹ️ Information"])
    
    with tab1:
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
                        st.image(image, caption=uploaded_file.name, use_container_width=True)
    
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
                        
                        # NDVI analysis of stitched result
                        if enable_ndvi:
                            st.subheader("🌱 NDVI Analysis of Stitched Map")
                            
                            try:
                                # Initialize NDVI analyzer
                                ndvi_analyzer = NDVIAnalyzer()
                                
                                # Perform NDVI analysis on stitched result
                                with st.spinner("Calculating NDVI for stitched map..."):
                                    stitched_ndvi_result = ndvi_analyzer.process_image(result, band_type)
                                
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.write("**Stitched Map**")
                                    stitched_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                                    st.image(stitched_rgb, use_container_width=True)
                                
                                with col2:
                                    st.write("**NDVI Map**")
                                    st.image(stitched_ndvi_result["ndvi_visualization"], use_container_width=True)
                                
                                # Analysis metrics for stitched result
                                st.write("**Vegetation Analysis of Stitched Map**")
                                stitched_analysis = stitched_ndvi_result["vegetation_analysis"]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean NDVI", f"{stitched_analysis['mean_ndvi']:.3f}")
                                    st.metric("Total Vegetation", f"{stitched_analysis['vegetation_coverage']:.1f}%")
                                with col2:
                                    st.metric("Dense Vegetation", f"{stitched_analysis['dense_vegetation_percentage']:.1f}%")
                                    st.metric("Moderate Vegetation", f"{stitched_analysis['moderate_vegetation_percentage']:.1f}%")
                                with col3:
                                    st.metric("Sparse Vegetation", f"{stitched_analysis['sparse_vegetation_percentage']:.1f}%")
                                    st.metric("Bare Soil", f"{stitched_analysis['bare_soil_percentage']:.1f}%")
                                
                                # Download NDVI map
                                st.write("**Download NDVI Analysis**")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    stitched_ndvi_pil = Image.fromarray(stitched_ndvi_result["ndvi_visualization"])
                                    stitched_ndvi_download = create_download_link(stitched_ndvi_pil, "PNG")
                                    st.markdown(stitched_ndvi_download.replace("Stitched Map", "NDVI Map"), unsafe_allow_html=True)
                                
                                with col2:
                                    stitched_comparison_plot = ndvi_analyzer.create_comparison_plot(result, stitched_ndvi_result)
                                    stitched_comparison_pil = Image.open(io.BytesIO(stitched_comparison_plot))
                                    stitched_comparison_download = create_download_link(stitched_comparison_pil, "PNG")
                                    st.markdown(stitched_comparison_download.replace("Stitched Map", "NDVI Analysis Report"), unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.warning(f"NDVI analysis of stitched result failed: {str(e)}")
                                st.info("You can still analyze individual images in the NDVI Analysis tab.")
                        
                    else:
                        st.error("❌ Stitching failed. Please try with different images or adjust settings.")
                        st.info("Tips:\n- Ensure images have sufficient overlap\n- Try adjusting the match threshold\n- Use images from the same height/angle")
                        
                except Exception as e:
                    st.error(f"An error occurred during stitching: {str(e)}")
                    st.info("Please check your images and try again with different settings.")
            else:
                st.info("Please upload at least 2 images to start stitching.")
    
    with tab2:
        st.header("🌱 NDVI Analysis")
        
        if uploaded_files:
            if enable_ndvi:
                st.info("NDVI analysis will be performed on both individual images and the final stitched result.")
                
                # Single image NDVI analysis
                st.subheader("Individual Image Analysis")
                
                selected_image_idx = st.selectbox(
                    "Select image for NDVI analysis",
                    range(len(uploaded_files)),
                    format_func=lambda x: uploaded_files[x].name
                )
                
                if st.button("🔬 Analyze Selected Image"):
                    try:
                        # Process selected image
                        uploaded_file = uploaded_files[selected_image_idx]
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        if validate_image(image_bytes):
                            # Convert to OpenCV format
                            image = np.array(Image.open(io.BytesIO(image_bytes)))
                            if len(image.shape) == 3:
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            
                            # Preprocess image
                            processed_image = preprocess_image(image)
                            
                            # Initialize NDVI analyzer
                            ndvi_analyzer = NDVIAnalyzer()
                            
                            # Perform NDVI analysis
                            with st.spinner("Calculating NDVI..."):
                                ndvi_result = ndvi_analyzer.process_image(processed_image, band_type)
                            
                            # Display results
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.subheader("Original Image")
                                original_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                                st.image(original_rgb, use_container_width=True)
                            
                            with col2:
                                st.subheader("NDVI Visualization")
                                st.image(ndvi_result["ndvi_visualization"], use_container_width=True)
                            
                            # Analysis metrics
                            st.subheader("Vegetation Analysis")
                            analysis = ndvi_result["vegetation_analysis"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean NDVI", f"{analysis['mean_ndvi']:.3f}")
                                st.metric("Vegetation Coverage", f"{analysis['vegetation_coverage']:.1f}%")
                            with col2:
                                st.metric("Dense Vegetation", f"{analysis['dense_vegetation_percentage']:.1f}%")
                                st.metric("Moderate Vegetation", f"{analysis['moderate_vegetation_percentage']:.1f}%")
                            with col3:
                                st.metric("Sparse Vegetation", f"{analysis['sparse_vegetation_percentage']:.1f}%")
                                st.metric("Bare Soil", f"{analysis['bare_soil_percentage']:.1f}%")
                            with col4:
                                st.metric("Water Bodies", f"{analysis['water_percentage']:.1f}%")
                                st.metric("NDVI Range", f"{analysis['min_ndvi']:.2f} to {analysis['max_ndvi']:.2f}")
                            
                            # NDVI histogram and detailed analysis
                            st.subheader("Detailed Analysis")
                            
                            # Create comparison plot
                            comparison_plot = ndvi_analyzer.create_comparison_plot(processed_image, ndvi_result)
                            st.image(comparison_plot, use_container_width=True)
                            
                            # NDVI legend
                            st.subheader("NDVI Color Scale")
                            legend_plot = ndvi_analyzer.get_ndvi_legend()
                            st.image(legend_plot, use_container_width=True)
                            
                            # Download options
                            st.subheader("Download Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download NDVI visualization
                                ndvi_pil = Image.fromarray(ndvi_result["ndvi_visualization"])
                                ndvi_download = create_download_link(ndvi_pil, "PNG")
                                st.markdown(ndvi_download, unsafe_allow_html=True)
                            
                            with col2:
                                # Download comparison plot
                                comparison_pil = Image.open(io.BytesIO(comparison_plot))
                                comparison_download = create_download_link(comparison_pil, "PNG")
                                st.markdown(comparison_download.replace("Stitched Map", "NDVI Analysis"), unsafe_allow_html=True)
                        
                        else:
                            st.error("Invalid image selected for NDVI analysis")
                    
                    except Exception as e:
                        st.error(f"NDVI analysis failed: {str(e)}")
                        st.info("Please check your image and try with different band configuration.")
                
                # Information about NDVI
                with st.expander("About NDVI Analysis"):
                    st.markdown("""
                    ### What is NDVI?
                    
                    NDVI (Normalized Difference Vegetation Index) is a widely used indicator of vegetation health and density:
                    
                    **Formula:** NDVI = (NIR - Red) / (NIR + Red)
                    
                    ### NDVI Value Ranges:
                    - **-1.0 to -0.3**: Water bodies, clouds, snow
                    - **-0.3 to 0.1**: Bare soil, rock, sand
                    - **0.1 to 0.3**: Sparse vegetation, stressed plants
                    - **0.3 to 0.6**: Moderate vegetation, healthy crops
                    - **0.6 to 1.0**: Dense, healthy vegetation, forests
                    
                    ### Band Configurations:
                    - **Standard RGB**: Uses Red and Green channels (Green as pseudo-NIR)
                    - **Modified RGB**: Uses Red and Blue channels (Blue as pseudo-NIR)
                    - **Infrared RGB**: For cameras with NIR capability (NIR in Red channel)
                    
                    ### Applications:
                    - Crop health monitoring
                    - Forest management
                    - Environmental monitoring
                    - Precision agriculture
                    """)
            else:
                st.info("Enable NDVI Analysis in the sidebar to analyze vegetation health in your drone images.")
        else:
            st.info("Upload drone images in the 'Upload & Stitch' tab to begin NDVI analysis.")
    
    with tab3:
        # Information section
        st.markdown("""
        ### Image Stitching Process
        
        1. **Feature Detection**: The algorithm detects key features in each image using SIFT or ORB detectors
        2. **Feature Matching**: Features are matched between overlapping images
        3. **Homography Estimation**: The geometric transformation between images is calculated
        4. **Image Warping**: Images are warped and aligned based on the homography
        5. **Blending**: Overlapping regions are blended to create a seamless result
        
        ### NDVI Analysis Process
        
        1. **Band Extraction**: Separates Red and Near-Infrared (NIR) bands from the image
        2. **NDVI Calculation**: Computes the normalized difference vegetation index
        3. **Visualization**: Creates colored maps showing vegetation health
        4. **Analysis**: Provides statistics on vegetation coverage and health
        
        ### Tips for Best Results
        
        **Image Stitching:**
        - Ensure images have 20-50% overlap
        - Use images taken from similar heights and angles
        - Avoid images with too much motion blur
        - Sequential images work better than random collections
        
        **NDVI Analysis:**
        - Use images with good lighting conditions
        - Avoid shadows and extreme lighting
        - For best results, use cameras with NIR capability
        - Standard RGB cameras provide approximate NDVI values
        """)

if __name__ == "__main__":
    main()
