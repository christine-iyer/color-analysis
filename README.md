# Color Analysis Tool

This project provides tools for extracting color palettes from images using multiple methods.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python color_analyzer.py
```

Or for a quick analysis:
```bash
python quick_analysis.py
```

## Methods Available

### 1. K-means Clustering
Uses machine learning to cluster similar colors and extract the most dominant ones. Best for getting representative colors.

### 2. ColorThief
Specialized library for palette extraction. Fast and effective for most images.

### 3. Most Common Colors
Analyzes color frequency with similarity tolerance. Good for images with distinct color regions.

## Features

- Extract 3, 5, or 8 dominant colors
- Multiple extraction algorithms
- Color name identification
- RGB and Hex color codes
- Visual palette display
- Results saved to text files

## Output

The tool will:
- Display color palettes visually using matplotlib
- Show RGB values, hex codes, and color names
- Save analysis results to text files
- Compare different extraction methods

## File Structure

- `color_analyzer.py` - Main analysis tool with all methods
- `quick_analysis.py` - Simple script for quick color extraction
- `requirements.txt` - Python dependencies
- `image.jpg` - Your image to analyze
