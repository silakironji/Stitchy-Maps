# Drone Image Stitcher & NDVI Analyzer

## Overview

This is a Streamlit-based web application that allows users to upload multiple drone images and stitch them together to create cohesive maps using advanced computer vision algorithms. The application also includes comprehensive NDVI (Normalized Difference Vegetation Index) analysis capabilities for vegetation health monitoring. The application leverages OpenCV for image processing, matplotlib for NDVI visualization, and provides an intuitive tabbed interface for both stitching and analysis features.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework
- **Interface**: Tabbed single-page application with three main sections:
  - Upload & Stitch: Image stitching functionality
  - NDVI Analysis: Vegetation health analysis tools
  - Information: Help and guidance documentation
- **Layout**: Wide layout configuration for better image display
- **Components**: File uploader, parameter controls, progress indicators, metrics display, and download functionality

### Backend Architecture
- **Core Logic**: Python-based image processing and analysis pipeline
- **Image Processing**: OpenCV library for computer vision operations
- **Scientific Analysis**: Matplotlib for NDVI visualization and data plotting
- **Database Layer**: PostgreSQL with SQLAlchemy ORM for data persistence
- **Modular Design**: Separated concerns across multiple modules:
  - `app.py`: Main Streamlit application and UI logic with tabbed interface
  - `image_stitcher.py`: Core stitching algorithms and feature detection
  - `ndvi_analyzer.py`: NDVI calculation, visualization, and vegetation health analysis
  - `database.py`: Database models, connections, and data management operations
  - `utils.py`: Helper functions for validation, preprocessing, and file operations

## Key Components

### 1. ImageStitcher Class
- **Purpose**: Handles the core image stitching functionality
- **Feature Detectors**: Supports both SIFT and ORB algorithms
- **Configurable Parameters**: Match threshold and RANSAC threshold for fine-tuning
- **Matcher**: Uses appropriate matchers based on the selected detector type

### 2. Utility Functions
- **Image Validation**: Ensures uploaded files are valid images using PIL
- **Preprocessing**: Resizes large images to optimize processing performance
- **Download Management**: Creates download links for processed results

### 3. NDVIAnalyzer Class
- **Purpose**: Handles NDVI calculation and vegetation health analysis
- **Band Processing**: Supports multiple band configurations (standard RGB, modified RGB, infrared RGB)
- **Visualization**: Creates colored NDVI maps with customizable colormaps
- **Analysis Metrics**: Provides comprehensive vegetation statistics and health indicators

### 4. Database Management System
- **Purpose**: Persistent storage and tracking of analysis sessions and results
- **Session Management**: Create, track, and manage analysis sessions with metadata
- **Result Storage**: Store stitching results, NDVI analyses, and processing metrics
- **Analytics**: Provide comprehensive analytics and historical data analysis
- **Export Capabilities**: Export session data and complete database exports

### 5. Streamlit Interface
- **Parameter Controls**: Sidebar with detector selection, threshold adjustments, and NDVI configuration
- **Tabbed Navigation**: Four-tab interface (Upload & Stitch, NDVI Analysis, Database, Information)
- **Session Management**: Create and track analysis sessions with persistent storage
- **Real-time Configuration**: Sliders and select boxes for interactive parameter tuning
- **User Guidance**: Help text, tooltips, and comprehensive information sections

## Data Flow

### Image Stitching Pipeline
1. **Image Upload**: Users upload multiple drone images through Streamlit file uploader
2. **Validation**: Each image is validated for format and integrity
3. **Preprocessing**: Images are resized if they exceed maximum dimensions (1500px)
4. **Feature Detection**: Keypoints and descriptors are extracted using selected algorithm
5. **Feature Matching**: Images are matched based on common features
6. **Stitching**: Images are aligned and blended to create final panorama
7. **Output**: Processed result is made available for download

### NDVI Analysis Pipeline
1. **Image Selection**: Users can analyze individual uploaded images or the final stitched result
2. **Band Extraction**: Red and NIR bands are extracted based on selected configuration
3. **NDVI Calculation**: Normalized Difference Vegetation Index is computed using the formula (NIR - Red) / (NIR + Red)
4. **Visualization**: NDVI values are mapped to colors using matplotlib colormaps
5. **Health Analysis**: Vegetation coverage, density, and health metrics are calculated
6. **Reporting**: Visual comparisons, histograms, and statistical reports are generated
7. **Export**: NDVI maps and analysis reports are made available for download

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **OpenCV (cv2)**: Computer vision library for image processing and stitching
- **NumPy**: Numerical computing for array operations and NDVI calculations
- **PIL (Pillow)**: Image manipulation and validation
- **Matplotlib**: Scientific plotting library for NDVI visualization and analysis charts
- **SQLAlchemy**: Object-Relational Mapping (ORM) for database operations
- **PostgreSQL**: Relational database for persistent data storage
- **Pandas**: Data manipulation and analysis for export capabilities

### Algorithms
- **SIFT**: Scale-Invariant Feature Transform for robust feature detection
- **ORB**: Oriented FAST and Rotated BRIEF for fast feature detection
- **RANSAC**: Random Sample Consensus for outlier detection in homography estimation
- **NDVI**: Normalized Difference Vegetation Index for vegetation health analysis
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for image enhancement

## Deployment Strategy

### Development Environment
- **Platform**: Replit-compatible Python environment
- **Dependencies**: Requirements managed through standard Python package management
- **Configuration**: Streamlit page configuration for optimal display

### Runtime Considerations
- **Memory Management**: Images are preprocessed to manage memory usage
- **Performance Optimization**: Configurable parameters allow users to balance quality vs speed
- **Error Handling**: Validation functions prevent processing of invalid inputs

### Scalability Notes
- Application processes images sequentially
- Memory usage scales with image size and quantity
- Processing time depends on image resolution and algorithm selection
- Consider implementing batch processing limits for production use

## Technical Architecture Decisions

### Feature Detection Choice
- **Problem**: Need reliable feature matching across drone images with varying perspectives
- **Solution**: Support for both SIFT (accuracy) and ORB (speed) algorithms
- **Rationale**: SIFT provides better accuracy for complex scenes, while ORB offers faster processing

### Preprocessing Strategy
- **Problem**: Large drone images can cause memory issues and slow processing
- **Solution**: Automatic resizing with configurable maximum dimensions
- **Trade-off**: Slight quality reduction for significantly improved performance

### Modular Design
- **Problem**: Maintainability and separation of concerns
- **Solution**: Split functionality across dedicated modules
- **Benefits**: Easier testing, debugging, and future enhancements

### Parameter Configurability
- **Problem**: Different image sets may require different stitching parameters
- **Solution**: Interactive controls for key algorithm parameters
- **User Experience**: Allows fine-tuning without code modification