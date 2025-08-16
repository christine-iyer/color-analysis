#!/usr/bin/env python3
"""
Interactive color analysis with manual substitution options.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import json

class InteractiveColorAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        
    def extract_comprehensive_palette(self, n_clusters=15):
        """Extract a comprehensive set of colors from the image."""
        # Resize for processing
        resized_image = self.image.resize((150, int(150 * self.image.height / self.image.width)))
        data = np.array(resized_image).reshape((-1, 3)).astype(np.float64)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(data)
        
        colors = kmeans.cluster_centers_.round(0).astype(int)
        return [tuple(int(c) for c in color) for color in colors]
    
    def categorize_colors(self, colors):
        """Categorize colors into different types."""
        categories = {
            'grays': [],
            'darks': [],
            'pinks': [],
            'browns': [],
            'golds': [],
            'greens': [],
            'others': []
        }
        
        for color in colors:
            r, g, b = color
            
            # Check if grayscale
            if abs(r - g) <= 25 and abs(g - b) <= 25 and abs(r - b) <= 25:
                categories['grays'].append(color)
            # Check if dark
            elif sum(color) / 3 < 60:
                categories['darks'].append(color)
            # Check if pink (improved detection)
            elif self.is_pink_tone(color):
                categories['pinks'].append(color)
            # Check if brown/tan
            elif r > g and g > b and r > 100:
                categories['browns'].append(color)
            # Check if gold/yellow
            elif r > 150 and g > 150 and b < 120:
                categories['golds'].append(color)
            # Check if green
            elif g > r and g > b:
                categories['greens'].append(color)
            else:
                categories['others'].append(color)
        
        return categories
    
    def is_pink_tone(self, rgb_color):
        """Detect pink tones using HSV color space."""
        r, g, b = [x/255.0 for x in rgb_color]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        if diff == 0:
            return False
            
        # Calculate hue
        if max_val == r:
            hue = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            hue = (60 * ((b - r) / diff) + 120) % 360
        else:
            hue = (60 * ((r - g) / diff) + 240) % 360
        
        # Pink hue ranges and criteria
        is_pink_hue = (300 <= hue <= 360) or (0 <= hue <= 30)
        has_red_dominance = rgb_color[0] >= rgb_color[1] and rgb_color[0] >= rgb_color[2]
        sufficient_brightness = max_val > 0.3
        
        return is_pink_hue and has_red_dominance and sufficient_brightness
    
    def create_custom_palette(self, exclude_categories=None, include_categories=None, 
                            manual_colors=None, palette_size=5):
        """Create a custom palette with specified criteria."""
        if exclude_categories is None:
            exclude_categories = ['grays', 'darks']
        if include_categories is None:
            include_categories = ['pinks', 'browns', 'golds', 'others']
        if manual_colors is None:
            manual_colors = []
        
        # Get comprehensive color set
        all_colors = self.extract_comprehensive_palette()
        categorized = self.categorize_colors(all_colors)
        
        # Build palette
        palette = []
        
        # Add manual colors first
        for color in manual_colors:
            if len(palette) < palette_size:
                palette.append(color)
        
        # Add colors from included categories
        for category in include_categories:
            for color in categorized.get(category, []):
                if len(palette) >= palette_size:
                    break
                # Avoid duplicates
                if not any(all(abs(color[i] - existing[i]) <= 30 for i in range(3)) 
                          for existing in palette):
                    palette.append(color)
        
        return palette[:palette_size], categorized
    
    def rgb_to_hex(self, rgb_color):
        return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])
    
    def get_color_name(self, rgb_color):
        """Get approximate color name."""
        css3_colors = {
            'black': (0, 0, 0), 'gray': (128, 128, 128), 'silver': (192, 192, 192),
            'white': (255, 255, 255), 'maroon': (128, 0, 0), 'red': (255, 0, 0),
            'olive': (128, 128, 0), 'yellow': (255, 255, 0), 'green': (0, 128, 0),
            'navy': (0, 0, 128), 'blue': (0, 0, 255), 'purple': (128, 0, 128),
            'teal': (0, 128, 128), 'aqua': (0, 255, 255), 'orange': (255, 165, 0),
            'brown': (165, 42, 42), 'pink': (255, 192, 203), 'gold': (255, 215, 0),
            'tan': (210, 180, 140), 'beige': (245, 245, 220), 'coral': (255, 127, 80),
            'salmon': (250, 128, 114), 'rose': (255, 0, 127), 'plum': (221, 160, 221)
        }
        
        min_distance = float('inf')
        closest_name = 'custom'
        
        for name, (r_c, g_c, b_c) in css3_colors.items():
            distance = sum((rgb_color[i] - (r_c, g_c, b_c)[i])**2 for i in range(3))**0.5
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name

def main():
    print("🎨 Interactive Color Analysis & Substitution Tool")
    print("=" * 50)
    
    analyzer = InteractiveColorAnalyzer("image.jpg")
    print(f"✅ Loaded image: {analyzer.image.size[0]}x{analyzer.image.size[1]} pixels")
    
    # Show all detected colors by category
    all_colors = analyzer.extract_comprehensive_palette(15)
    categorized = analyzer.categorize_colors(all_colors)
    
    print(f"\n📊 COLOR ANALYSIS BY CATEGORY:")
    print("=" * 35)
    
    for category, colors in categorized.items():
        if colors:
            print(f"\n{category.upper()}:")
            for i, color in enumerate(colors[:3]):  # Show top 3 per category
                hex_color = analyzer.rgb_to_hex(color)
                color_name = analyzer.get_color_name(color)
                print(f"   {i+1}. RGB{color} → {hex_color} ({color_name})")
    
    # Create different palette options
    print(f"\n🎯 PALETTE OPTIONS:")
    print("=" * 20)
    
    # Option 1: No grays or darks
    print(f"\n1️⃣  EXCLUDE GRAYS & DARKS:")
    palette1, _ = analyzer.create_custom_palette(
        exclude_categories=['grays', 'darks'],
        palette_size=5
    )
    for i, color in enumerate(palette1):
        hex_color = analyzer.rgb_to_hex(color)
        color_name = analyzer.get_color_name(color)
        is_pink = "🌸" if analyzer.is_pink_tone(color) else "🎨"
        print(f"   {i+1}. {is_pink} RGB{color} → {hex_color} ({color_name})")
    
    # Option 2: Prioritize pinks and warm tones
    print(f"\n2️⃣  PRIORITIZE PINKS & WARM TONES:")
    warm_manual_colors = [
        (210, 170, 180),  # Dusty rose
        (200, 160, 170),  # Muted pink
        (190, 140, 130),  # Warm taupe
    ]
    palette2, _ = analyzer.create_custom_palette(
        exclude_categories=['grays', 'darks'],
        include_categories=['pinks', 'browns', 'others'],
        manual_colors=warm_manual_colors,
        palette_size=5
    )
    for i, color in enumerate(palette2):
        hex_color = analyzer.rgb_to_hex(color)
        color_name = analyzer.get_color_name(color)
        is_pink = "🌸" if analyzer.is_pink_tone(color) else "🎨"
        print(f"   {i+1}. {is_pink} RGB{color} → {hex_color} ({color_name})")
    
    # Option 3: Custom pink-enhanced palette
    print(f"\n3️⃣  PINK-ENHANCED PALETTE:")
    pink_colors = [
        (220, 180, 190),  # Soft pink
        (200, 150, 160),  # Rose
        (180, 140, 150),  # Deep rose
    ]
    palette3, _ = analyzer.create_custom_palette(
        exclude_categories=['grays', 'darks'],
        manual_colors=pink_colors,
        palette_size=5
    )
    for i, color in enumerate(palette3):
        hex_color = analyzer.rgb_to_hex(color)
        color_name = analyzer.get_color_name(color)
        is_pink = "🌸" if analyzer.is_pink_tone(color) else "🎨"
        print(f"   {i+1}. {is_pink} RGB{color} → {hex_color} ({color_name})")
    
    # Save all options
    results = {
        'image_info': {'path': 'image.jpg', 'size': analyzer.image.size},
        'color_categories': {cat: [{'rgb': c, 'hex': analyzer.rgb_to_hex(c)} 
                                  for c in colors] for cat, colors in categorized.items()},
        'palette_options': {
            'no_grays_darks': [{'rgb': c, 'hex': analyzer.rgb_to_hex(c), 
                               'name': analyzer.get_color_name(c)} for c in palette1],
            'warm_tones': [{'rgb': c, 'hex': analyzer.rgb_to_hex(c), 
                           'name': analyzer.get_color_name(c)} for c in palette2],
            'pink_enhanced': [{'rgb': c, 'hex': analyzer.rgb_to_hex(c), 
                              'name': analyzer.get_color_name(c)} for c in palette3]
        }
    }
    
    with open('interactive_color_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: interactive_color_analysis.json")
    
    # Provide copy-paste ready formats
    print(f"\n🌐 COPY-PASTE READY PALETTES:")
    print("-" * 35)
    
    palettes = {
        "No Grays/Darks": palette1,
        "Warm Tones": palette2, 
        "Pink Enhanced": palette3
    }
    
    for name, palette in palettes.items():
        hex_colors = [analyzer.rgb_to_hex(c) for c in palette]
        print(f"\n{name}:")
        print(f"   CSS: {' '.join(hex_colors)}")
        print(f"   Array: [{', '.join([f'\"#{c[1:]}\"' for c in hex_colors])}]")

if __name__ == "__main__":
    main()
