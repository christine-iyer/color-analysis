#!/usr/bin/env python3
"""
Simple usage example for the color analyzer.
Run this after installing dependencies with: pip install -r requirements.txt
"""

from color_analyzer import ColorAnalyzer

def quick_analysis():
    """Quick color analysis of the image."""
    print("🎨 Color Analysis Tool")
    print("=" * 30)
    
    # Initialize analyzer
    analyzer = ColorAnalyzer("image.jpg")
    
    # Quick analysis with 5 colors using K-means
    print("Extracting 5 dominant colors using K-means clustering...")
    colors = analyzer.extract_palette_kmeans(5)
    
    print("\nDominant Colors Found:")
    print("-" * 20)
    for i, color in enumerate(colors):
        hex_color = analyzer.rgb_to_hex(color)
        try:
            color_name = analyzer.get_color_name(color)
            print(f"{i+1}. RGB{color} | {hex_color} | {color_name}")
        except:
            print(f"{i+1}. RGB{color} | {hex_color}")
    
    # Display the palette
    analyzer.display_palette(colors, "Quick Analysis - K-means")

if __name__ == "__main__":
    quick_analysis()
