#!/usr/bin/env python3
"""
Simple color extraction without matplotlib visualization.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import colorthief
import json

class SimpleColorAnalyzer:
    def __init__(self, image_path):
        """Initialize the color analyzer with an image path."""
        self.image_path = image_path
        self.image = Image.open(image_path)
        
    def extract_palette_kmeans(self, n_colors=5, resize_width=150):
        """Extract color palette using K-means clustering."""
        # Resize image for faster processing
        aspect_ratio = self.image.height / self.image.width
        new_height = int(resize_width * aspect_ratio)
        resized_image = self.image.resize((resize_width, new_height))
        
        # Convert to numpy array and reshape
        data = np.array(resized_image)
        data = data.reshape((-1, 3))
        data = data.astype(np.float64)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(data)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        colors = colors.round(0).astype(int)
        
        return [tuple(int(c) for c in color) for color in colors]
    
    def extract_palette_colorthief(self, n_colors=5):
        """Extract color palette using ColorThief library."""
        color_thief = colorthief.ColorThief(self.image_path)
        
        if n_colors > 1:
            palette = color_thief.get_palette(color_count=n_colors, quality=1)
            return palette
        else:
            dominant_color = color_thief.get_color(quality=1)
            return [dominant_color]
    
    def extract_most_common_colors(self, n_colors=5, tolerance=10):
        """Extract most common colors by frequency."""
        if self.image.mode != 'RGB':
            rgb_image = self.image.convert('RGB')
        else:
            rgb_image = self.image
            
        # Get all pixel colors
        pixels = list(rgb_image.getdata())
        
        # Group similar colors
        grouped_colors = {}
        for pixel in pixels:
            found_group = False
            for group_color in grouped_colors:
                if all(abs(pixel[i] - group_color[i]) <= tolerance for i in range(3)):
                    grouped_colors[group_color] += 1
                    found_group = True
                    break
            if not found_group:
                grouped_colors[pixel] = 1
        
        # Get most common colors
        most_common = Counter(grouped_colors).most_common(n_colors)
        return [color for color, count in most_common]
    
    def rgb_to_hex(self, rgb_color):
        """Convert RGB to hex format."""
        return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])
    
    def get_color_name(self, rgb_color):
        """Get the closest color name for an RGB value."""
        css3_colors = {
            'black': (0, 0, 0), 'silver': (192, 192, 192), 'gray': (128, 128, 128),
            'white': (255, 255, 255), 'maroon': (128, 0, 0), 'red': (255, 0, 0),
            'purple': (128, 0, 128), 'fuchsia': (255, 0, 255), 'green': (0, 128, 0),
            'lime': (0, 255, 0), 'olive': (128, 128, 0), 'yellow': (255, 255, 0),
            'navy': (0, 0, 128), 'blue': (0, 0, 255), 'teal': (0, 128, 128),
            'aqua': (0, 255, 255), 'orange': (255, 165, 0), 'brown': (165, 42, 42),
            'pink': (255, 192, 203), 'gold': (255, 215, 0), 'beige': (245, 245, 220),
            'tan': (210, 180, 140), 'khaki': (240, 230, 140), 'violet': (238, 130, 238)
        }
        
        min_distance = float('inf')
        closest_name = 'unknown'
        
        for name, (r_c, g_c, b_c) in css3_colors.items():
            distance = ((r_c - rgb_color[0]) ** 2 + 
                       (g_c - rgb_color[1]) ** 2 + 
                       (b_c - rgb_color[2]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name

def main():
    """Main analysis function."""
    print("🎨 Simple Color Analysis Tool")
    print("=" * 35)
    
    try:
        analyzer = SimpleColorAnalyzer("image.jpg")
        print(f"✅ Loaded image: {analyzer.image.size[0]}x{analyzer.image.size[1]} pixels")
        
        # Method 1: K-means clustering
        print("\n1️⃣  K-means Clustering Analysis:")
        print("-" * 30)
        kmeans_colors = analyzer.extract_palette_kmeans(5)
        for i, color in enumerate(kmeans_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            print(f"   {i+1}. RGB{color} → {hex_color} ({color_name})")
        
        # Method 2: ColorThief
        print("\n2️⃣  ColorThief Analysis:")
        print("-" * 30)
        try:
            colorthief_colors = analyzer.extract_palette_colorthief(5)
            for i, color in enumerate(colorthief_colors):
                hex_color = analyzer.rgb_to_hex(color)
                color_name = analyzer.get_color_name(color)
                print(f"   {i+1}. RGB{color} → {hex_color} ({color_name})")
        except Exception as e:
            print(f"   ❌ ColorThief failed: {e}")
        
        # Method 3: Most common colors
        print("\n3️⃣  Most Common Colors Analysis:")
        print("-" * 30)
        try:
            common_colors = analyzer.extract_most_common_colors(5)
            for i, color in enumerate(common_colors):
                hex_color = analyzer.rgb_to_hex(color)
                color_name = analyzer.get_color_name(color)
                print(f"   {i+1}. RGB{color} → {hex_color} ({color_name})")
        except Exception as e:
            print(f"   ❌ Most common colors failed: {e}")
        
        # Save results to JSON
        results = {
            'image_info': {
                'path': 'image.jpg',
                'size': analyzer.image.size,
                'mode': analyzer.image.mode
            },
            'analysis_results': {
                'kmeans': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color)
                    } for color in kmeans_colors
                ],
                'colorthief': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color)
                    } for color in (colorthief_colors if 'colorthief_colors' in locals() else [])
                ],
                'most_common': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color)
                    } for color in (common_colors if 'common_colors' in locals() else [])
                ]
            }
        }
        
        with open('color_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: color_analysis_results.json")
        print(f"\n🎯 RECOMMENDED PALETTE (K-means method):")
        for i, color in enumerate(kmeans_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            print(f"   • {hex_color} ({color_name})")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
