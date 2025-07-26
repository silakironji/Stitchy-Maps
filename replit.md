# Drone Image Stitcher

## Overview

This is a Streamlit-based web application that allows users to upload multiple drone images and stitch them together to create cohesive maps using advanced computer vision algorithms. The application leverages OpenCV for image processing and provides an intuitive interface for configuring stitching parameters.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework
- **Interface**: Single-page application with sidebar controls
- **Layout**: Wide layout configuration for better image display
- **Components**: File uploader, parameter controls, progress indicators, and download functionality

### Backend Architecture
- **Core Logic**: Python-based image processing pipeline
- **Image Processing**: OpenCV library for computer vision operations
- **Modular Design**: Separated concerns across multiple modules:
  - `app.py`: Main Streamlit application and UI logic
  - `image_stitcher.py`: Core stitching algorithms and feature detection
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

### 3. Streamlit Interface
- **Parameter Controls**: Sidebar with detector selection and threshold adjustments
- **Real-time Configuration**: Sliders and select boxes for interactive parameter tuning
- **User Guidance**: Help text and tooltips for each configuration option

## Data Flow

1. **Image Upload**: Users upload multiple drone images through Streamlit file uploader
2. **Validation**: Each image is validated for format and integrity
3. **Preprocessing**: Images are resized if they exceed maximum dimensions (1500px)
4. **Feature Detection**: Keypoints and descriptors are extracted using selected algorithm
5. **Feature Matching**: Images are matched based on common features
6. **Stitching**: Images are aligned and blended to create final panorama
7. **Output**: Processed result is made available for download

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **OpenCV (cv2)**: Computer vision library for image processing and stitching
- **NumPy**: Numerical computing for array operations
- **PIL (Pillow)**: Image manipulation and validation

### Algorithms
- **SIFT**: Scale-Invariant Feature Transform for robust feature detection
- **ORB**: Oriented FAST and Rotated BRIEF for fast feature detection
- **RANSAC**: Random Sample Consensus for outlier detection in homography estimation

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