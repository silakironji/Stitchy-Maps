import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import time
from image_stitcher import ImageStitcher
from utils import validate_image, preprocess_image, create_download_link
from ndvi_analyzer import NDVIAnalyzer
from database import db_manager, initialize_database

def main():
    st.set_page_config(
        page_title="Drone Image Stitcher & NDVI Analyzer",
        page_icon="🗺️",
        layout="wide"
    )
    
    # Initialize database
    if 'db_initialized' not in st.session_state:
        with st.spinner("Initializing database..."):
            if initialize_database():
                st.session_state.db_initialized = True
            else:
                st.error("Failed to initialize database. Some features may not work properly.")
                st.session_state.db_initialized = False
    
    # Initialize session state
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'session_name' not in st.session_state:
        st.session_state.session_name = ""
    
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
    
    # Session management in sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Session Management")
        
        # Session name input
        session_name = st.text_input(
            "Session Name (Optional)",
            value=st.session_state.session_name,
            help="Give your analysis session a memorable name"
        )
        
        if session_name != st.session_state.session_name:
            st.session_state.session_name = session_name
        
        # Start new session button
        if st.button("🆕 Start New Session"):
            if uploaded_files:
                # Create new session in database
                session_id = db_manager.create_analysis_session(
                    session_name=session_name if session_name else None,
                    total_images=len(uploaded_files),
                    detector_type=detector_type,
                    match_threshold=match_threshold,
                    ransac_threshold=ransac_threshold,
                    ndvi_enabled=enable_ndvi,
                    band_type=band_type if enable_ndvi else "standard_rgb"
                )
                
                if session_id:
                    st.session_state.current_session_id = session_id
                    st.success(f"Started new session: {session_id[:8]}...")
                    
                    # Save image metadata
                    for uploaded_file in uploaded_files:
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        if validate_image(image_bytes):
                            image = Image.open(io.BytesIO(image_bytes))
                            db_manager.save_image_metadata(
                                session_id=session_id,
                                filename=uploaded_file.name,
                                file_size=len(image_bytes),
                                width=image.width,
                                height=image.height
                            )
                else:
                    st.error("Failed to create session")
            else:
                st.warning("Please upload images first")
        
        # Current session info
        if st.session_state.current_session_id:
            st.info(f"Current Session: {st.session_state.current_session_id[:8]}...")

    # Main content area - use tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Stitch", "🌱 NDVI Analysis", "📊 Database", "ℹ️ Information"])
    
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
                start_time = time.time()
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
                        st.image(result_pil, caption="Stitched Drone Map", use_container_width=True)
                        
                        # Create download link
                        download_link = create_download_link(result_pil, output_format)
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Display image info
                        st.info(f"Result dimensions: {result.shape[1]} x {result.shape[0]} pixels")
                        
                        # Save to database
                        processing_time = time.time() - start_time
                        if st.session_state.current_session_id:
                            db_manager.save_stitching_result(
                                session_id=st.session_state.current_session_id,
                                result_width=result.shape[1],
                                result_height=result.shape[0],
                                processing_time=processing_time,
                                success=True,
                                settings={
                                    'detector_type': detector_type,
                                    'match_threshold': match_threshold,
                                    'ransac_threshold': ransac_threshold,
                                    'output_format': output_format
                                }
                            )
                            db_manager.update_session(st.session_state.current_session_id, stitching_completed=True)
                        
                        # NDVI analysis of stitched result
                        if enable_ndvi:
                            st.subheader("🌱 NDVI Analysis of Stitched Map")
                            
                            try:
                                # Initialize NDVI analyzer
                                ndvi_analyzer = NDVIAnalyzer()
                                
                                # Perform NDVI analysis on stitched result
                                with st.spinner("Calculating NDVI for stitched map..."):
                                    ndvi_start_time = time.time()
                                    stitched_ndvi_result = ndvi_analyzer.process_image(result, band_type)
                                    ndvi_processing_time = time.time() - ndvi_start_time
                                    
                                    # Save NDVI analysis to database
                                    if st.session_state.current_session_id:
                                        db_manager.save_ndvi_analysis(
                                            session_id=st.session_state.current_session_id,
                                            image_type="stitched",
                                            analysis_data=stitched_ndvi_result["vegetation_analysis"],
                                            processing_time=ndvi_processing_time,
                                            band_configuration=band_type,
                                            image_filename=None
                                        )
                                
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
                        processing_time = time.time() - start_time
                        st.error("❌ Stitching failed. Please try with different images or adjust settings.")
                        st.info("Tips:\n- Ensure images have sufficient overlap\n- Try adjusting the match threshold\n- Use images from the same height/angle")
                        
                        # Save failed attempt to database
                        if st.session_state.current_session_id:
                            db_manager.save_stitching_result(
                                session_id=st.session_state.current_session_id,
                                result_width=0,
                                result_height=0,
                                processing_time=processing_time,
                                success=False,
                                settings={
                                    'detector_type': detector_type,
                                    'match_threshold': match_threshold,
                                    'ransac_threshold': ransac_threshold,
                                    'output_format': output_format
                                },
                                error_message="Stitching failed - insufficient matches or overlap"
                            )
                        
                except Exception as e:
                    processing_time = time.time() - start_time
                    st.error(f"An error occurred during stitching: {str(e)}")
                    st.info("Please check your images and try again with different settings.")
                    
                    # Save error to database
                    if st.session_state.current_session_id:
                        db_manager.save_stitching_result(
                            session_id=st.session_state.current_session_id,
                            result_width=0,
                            result_height=0,
                            processing_time=processing_time,
                            success=False,
                            settings={
                                'detector_type': detector_type,
                                'match_threshold': match_threshold,
                                'ransac_threshold': ransac_threshold,
                                'output_format': output_format
                            },
                            error_message=str(e)
                        )
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
                                ndvi_start_time = time.time()
                                ndvi_result = ndvi_analyzer.process_image(processed_image, band_type)
                                ndvi_processing_time = time.time() - ndvi_start_time
                                
                                # Save NDVI analysis to database
                                if st.session_state.current_session_id:
                                    db_manager.save_ndvi_analysis(
                                        session_id=st.session_state.current_session_id,
                                        image_type="individual",
                                        analysis_data=ndvi_result["vegetation_analysis"],
                                        processing_time=ndvi_processing_time,
                                        band_configuration=band_type,
                                        image_filename=uploaded_files[selected_image_idx].name
                                    )
                            
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
        st.header("📊 Database Management")
        
        # Analytics Summary
        st.subheader("📈 Analytics Summary")
        
        if st.button("🔄 Refresh Analytics"):
            analytics = db_manager.get_analytics_summary()
            
            if analytics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Sessions", analytics.get('total_sessions', 0))
                    st.metric("Images Processed", analytics.get('total_images_processed', 0))
                
                with col2:
                    st.metric("Successful Stitching", analytics.get('successful_stitching_operations', 0))
                    st.metric("NDVI Analyses", analytics.get('total_ndvi_analyses', 0))
                
                with col3:
                    st.metric("Avg Stitching Time", f"{analytics.get('average_stitching_time', 0):.1f}s")
                    st.metric("Avg NDVI Time", f"{analytics.get('average_ndvi_time', 0):.1f}s")
                
                with col4:
                    st.metric("Most Used Detector", analytics.get('most_used_detector', 'N/A'))
                
                # Detector usage chart
                if analytics.get('detector_usage'):
                    st.subheader("🔍 Detector Usage Distribution")
                    detector_df = pd.DataFrame(
                        list(analytics['detector_usage'].items()),
                        columns=['Detector', 'Count']
                    )
                    st.bar_chart(detector_df.set_index('Detector'))
        
        # Session History
        st.subheader("📋 Recent Sessions")
        
        session_history = db_manager.get_session_history(limit=20)
        
        if session_history:
            # Create DataFrame for display
            history_df = pd.DataFrame(session_history)
            history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Select columns for display
            display_columns = ['session_name', 'created_at', 'total_images', 'detector_type', 'stitching_completed', 'ndvi_enabled']
            display_df = history_df[display_columns].copy()
            display_df.columns = ['Session Name', 'Created', 'Images', 'Detector', 'Stitched', 'NDVI']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Session Details
            st.subheader("🔍 Session Details")
            
            # Session selection
            session_options = {f"{s['session_name'] or 'Unnamed'} ({s['session_id'][:8]})": s['session_id'] 
                             for s in session_history}
            
            if session_options:
                selected_session_display = st.selectbox(
                    "Select session to view details",
                    list(session_options.keys())
                )
                
                if selected_session_display:
                    selected_session_id = session_options[selected_session_display]
                    
                    if st.button("📋 Load Session Details"):
                        session_details = db_manager.get_session_details(selected_session_id)
                        
                        if session_details:
                            # Session info
                            session_info = session_details['session']
                            st.write("**Session Information:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"- **Name:** {session_info['session_name'] or 'Unnamed'}")
                                st.write(f"- **Created:** {session_info['created_at']}")
                                st.write(f"- **Total Images:** {session_info['total_images']}")
                                st.write(f"- **Detector:** {session_info['detector_type']}")
                            
                            with col2:
                                st.write(f"- **Match Threshold:** {session_info['settings']['match_threshold']}")
                                st.write(f"- **RANSAC Threshold:** {session_info['settings']['ransac_threshold']}")
                                st.write(f"- **Band Type:** {session_info['settings']['band_type']}")
                                st.write(f"- **NDVI Enabled:** {'Yes' if session_info['ndvi_enabled'] else 'No'}")
                            
                            # Images
                            if session_details['images']:
                                st.write("**Uploaded Images:**")
                                images_df = pd.DataFrame(session_details['images'])
                                images_df['upload_time'] = pd.to_datetime(images_df['upload_time']).dt.strftime('%H:%M:%S')
                                st.dataframe(images_df, use_container_width=True)
                            
                            # Stitching result
                            if session_details['stitching_result']:
                                st.write("**Stitching Result:**")
                                stitching = session_details['stitching_result']
                                if stitching['success']:
                                    st.success(f"✅ Success - Dimensions: {stitching['dimensions']}, Time: {stitching['processing_time']:.1f}s")
                                else:
                                    st.error(f"❌ Failed - {stitching['error_message'] or 'Unknown error'}")
                            
                            # NDVI analyses
                            if session_details['ndvi_analyses']:
                                st.write("**NDVI Analyses:**")
                                ndvi_df = pd.DataFrame(session_details['ndvi_analyses'])
                                ndvi_df['created_at'] = pd.to_datetime(ndvi_df['created_at']).dt.strftime('%H:%M:%S')
                                ndvi_df['mean_ndvi'] = ndvi_df['mean_ndvi'].round(3)
                                ndvi_df['vegetation_coverage'] = ndvi_df['vegetation_coverage'].round(1)
                                st.dataframe(ndvi_df, use_container_width=True)
                            
                            # Export session data
                            if st.button("📁 Export Session Data"):
                                export_df = db_manager.export_session_data(selected_session_id)
                                if export_df is not None and not export_df.empty:
                                    csv_data = export_df.to_csv(index=False)
                                    st.download_button(
                                        label="💾 Download CSV",
                                        data=csv_data,
                                        file_name=f"session_{selected_session_id[:8]}_data.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.warning("No data available for export")
                        else:
                            st.error("Failed to load session details")
        else:
            st.info("No sessions found. Start by creating a session in the 'Upload & Stitch' tab.")
        
        # Database Management
        st.subheader("🛠️ Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Check Database Status"):
                try:
                    # Test database connection
                    analytics = db_manager.get_analytics_summary()
                    if analytics is not None:
                        st.success("✅ Database connection successful")
                        st.info(f"Database contains {analytics.get('total_sessions', 0)} sessions")
                    else:
                        st.error("❌ Database connection failed")
                except Exception as e:
                    st.error(f"❌ Database error: {str(e)}")
        
        with col2:
            if st.button("📊 Export All Data"):
                try:
                    # Get all session data for export
                    all_sessions = db_manager.get_session_history(limit=1000)  # Get all sessions
                    
                    if all_sessions:
                        # Create comprehensive export
                        export_data = []
                        
                        for session in all_sessions:
                            session_details = db_manager.get_session_details(session['session_id'])
                            if session_details:
                                # Add session data
                                base_data = {
                                    'session_id': session['session_id'],
                                    'session_name': session['session_name'],
                                    'created_at': session['created_at'],
                                    'total_images': session['total_images'],
                                    'detector_type': session['detector_type'],
                                    'stitching_completed': session['stitching_completed'],
                                    'ndvi_enabled': session['ndvi_enabled']
                                }
                                
                                # Add NDVI data if available
                                if session_details['ndvi_analyses']:
                                    for analysis in session_details['ndvi_analyses']:
                                        row = base_data.copy()
                                        row.update({
                                            'analysis_type': 'ndvi',
                                            'image_type': analysis['image_type'],
                                            'mean_ndvi': analysis['mean_ndvi'],
                                            'vegetation_coverage': analysis['vegetation_coverage'],
                                            'processing_time': analysis['processing_time']
                                        })
                                        export_data.append(row)
                                else:
                                    export_data.append(base_data)
                        
                        if export_data:
                            export_df = pd.DataFrame(export_data)
                            csv_data = export_df.to_csv(index=False)
                            
                            st.download_button(
                                label="💾 Download Complete Database Export",
                                data=csv_data,
                                file_name=f"drone_analysis_database_export.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No data available for export")
                    else:
                        st.info("No sessions found in database")
                        
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")

    with tab4:
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
