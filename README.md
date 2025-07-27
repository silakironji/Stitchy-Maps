# Drone Image Stitcher & NDVI Analyzer

A powerful Streamlit web application for stitching drone images into cohesive maps and analyzing vegetation health using NDVI (Normalized Difference Vegetation Index).

## Features

### 🗺️ Image Stitching
- **Advanced Algorithms**: Support for SIFT and ORB feature detection
- **Configurable Parameters**: Adjustable matching and RANSAC thresholds
- **Multiple Formats**: Support for PNG, JPEG, TIFF, and BMP images
- **Progress Tracking**: Real-time progress updates during processing
- **Download Options**: Export stitched maps in PNG or JPEG format

### 🌱 NDVI Analysis
- **Vegetation Health Monitoring**: Calculate NDVI for individual images and stitched results
- **Multiple Band Configurations**: Support for standard RGB, modified RGB, and infrared RGB
- **Visual Analytics**: Colored NDVI maps with customizable colormaps
- **Comprehensive Metrics**: Detailed vegetation coverage and health statistics
- **Export Reports**: Download NDVI visualizations and analysis reports

## Installation

### Requirements
- Python 3.11+
- Streamlit
- OpenCV
- NumPy
- Pillow
- Matplotlib

### Setup
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd drone-image-stitcher
   ```

2. Install dependencies:
   ```bash
   pip install streamlit opencv-python numpy pillow matplotlib
   ```

3. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```

## Usage

### Image Stitching
1. Navigate to the "Upload & Stitch" tab
2. Upload 2 or more overlapping drone images
3. Adjust stitching parameters in the sidebar if needed
4. Click "Start Stitching" to process your images
5. Download the resulting stitched map

### NDVI Analysis
1. Upload drone images in the "Upload & Stitch" tab
2. Enable "NDVI Analysis" in the sidebar
3. Go to the "NDVI Analysis" tab
4. Select an image and click "Analyze Selected Image"
5. View vegetation health metrics and download results

## Technical Details

### Supported Image Formats
- PNG, JPEG, TIFF, BMP
- Automatic preprocessing for optimal performance
- Maximum dimension limit: 1500px (configurable)

### NDVI Calculation
NDVI = (NIR - Red) / (NIR + Red)

### NDVI Value Interpretation
- **-1.0 to -0.3**: Water bodies, clouds, snow
- **-0.3 to 0.1**: Bare soil, rock, sand
- **0.1 to 0.3**: Sparse vegetation, stressed plants
- **0.3 to 0.6**: Moderate vegetation, healthy crops
- **0.6 to 1.0**: Dense, healthy vegetation, forests

### Band Configurations
- **Standard RGB**: Red and Green channels (Green as pseudo-NIR)
- **Modified RGB**: Red and Blue channels (Blue as pseudo-NIR)
- **Infrared RGB**: For NIR-capable cameras (NIR in Red channel)

## Applications

- **Agriculture**: Crop health monitoring and precision farming
- **Environmental**: Forest management and conservation
- **Research**: Vegetation studies and ecological monitoring
- **Mapping**: Creating detailed aerial maps with vegetation analysis

## Tips for Best Results

### Image Stitching
- Ensure 20-50% overlap between images
- Use consistent height and angle
- Avoid motion blur
- Sequential capture works better than random collections

### NDVI Analysis
- Use good lighting conditions
- Avoid shadows and extreme lighting
- NIR-capable cameras provide best results
- Standard RGB cameras give approximate values

## Project Structure

```
├── app.py                 # Main Streamlit application
├── image_stitcher.py      # Core stitching algorithms
├── ndvi_analyzer.py       # NDVI calculation and analysis
├── utils.py              # Helper functions
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Feel free to use and modify for your needs.

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [Matplotlib](https://matplotlib.org/) - Scientific plotting library

---

**Note**: For production use with large datasets, consider implementing batch processing limits and memory optimization techniques.