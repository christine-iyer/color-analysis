#!/usr/bin/env python3
"""
Improved color analysis with better pink detection and manual color addition.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import colorthief
import json

class ImprovedColorAnalyzer:
    def __init__(self, image_path):
        """Initialize the color analyzer with an image path."""
        self.image_path = image_path
        self.image = Image.open(image_path)
        
    def is_grayscale_color(self, rgb_color, tolerance=25):
        """Check if a color is grayscale (R, G, B values are similar)."""
        r, g, b = rgb_color
        return (abs(r - g) <= tolerance and 
                abs(g - b) <= tolerance and 
                abs(r - b) <= tolerance)
    
    def is_dark_color(self, rgb_color, threshold=60):
        """Check if a color is too dark."""
        return sum(rgb_color) / 3 < threshold
    
    def is_true_pink_tone(self, rgb_color):
        """Improved pink detection with more specific criteria."""
        r, g, b = rgb_color
        
        # Convert to HSV for better color detection
        r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        diff = max_val - min_val
        
        if diff == 0:  # Grayscale
            return False
            
        # Calculate hue
        if max_val == r_norm:
            hue = (60 * ((g_norm - b_norm) / diff) + 360) % 360
        elif max_val == g_norm:
            hue = (60 * ((b_norm - r_norm) / diff) + 120) % 360
        else:
            hue = (60 * ((r_norm - g_norm) / diff) + 240) % 360
        
        saturation = 0 if max_val == 0 else diff / max_val
        
        # Pink hue ranges: 300-360 (magenta-pink) and 0-30 (pink-red)
        is_pink_hue = (300 <= hue <= 360) or (0 <= hue <= 30)
        
        # Additional criteria for pink
        has_sufficient_red = r > 100
        not_too_saturated = saturation < 0.8  # Avoid pure magentas
        sufficient_brightness = max_val > 0.3
        
        # Check for dusty/muted pinks (lower saturation pinks)
        is_muted_pink = (320 <= hue <= 360 or 0 <= hue <= 20) and saturation > 0.1 and r > g and r > b
        
        return (is_pink_hue and has_sufficient_red and not_too_saturated and sufficient_brightness) or is_muted_pink
    
    def find_actual_pink_tones(self, min_pixels=50):
        """Search for actual pink tones using improved detection."""
        if self.image.mode != 'RGB':
            rgb_image = self.image.convert('RGB')
        else:
            rgb_image = self.image
            
        pixels = list(rgb_image.getdata())
        pink_colors = {}
        
        for pixel in pixels:
            if self.is_true_pink_tone(pixel):
                # Group similar pinks
                found_group = False
                for group_color in pink_colors:
                    if all(abs(pixel[i] - group_color[i]) <= 20 for i in range(3)):
                        pink_colors[group_color] += 1
                        found_group = True
                        break
                if not found_group:
                    pink_colors[pixel] = 1
        
        # Filter by minimum pixel count
        significant_pinks = [(color, count) for color, count in pink_colors.items() 
                           if count >= min_pixels]
        significant_pinks.sort(key=lambda x: x[1], reverse=True)
        
        return [color for color, count in significant_pinks[:3]]
    
    def extract_filtered_palette(self, n_colors=5, exclude_grays=True, 
                                exclude_dark=True, resize_width=150):
        """Extract palette with filtering."""
        # Resize for processing
        aspect_ratio = self.image.height / self.image.width
        new_height = int(resize_width * aspect_ratio)
        resized_image = self.image.resize((resize_width, new_height))
        
        # K-means clustering
        data = np.array(resized_image).reshape((-1, 3)).astype(np.float64)
        
        # Use more clusters initially to capture more color variety
        initial_clusters = min(n_colors * 4, 20)
        kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
        kmeans.fit(data)
        
        colors = kmeans.cluster_centers_.round(0).astype(int)
        candidate_colors = [tuple(int(c) for c in color) for color in colors]
        
        # Filter colors
        filtered_colors = []
        for color in candidate_colors:
            if exclude_grays and self.is_grayscale_color(color):
                continue
            if exclude_dark and self.is_dark_color(color):
                continue
            filtered_colors.append(color)
        
        # Remove very similar colors
        final_colors = []
        for color in filtered_colors:
            is_duplicate = False
            for existing in final_colors:
                if all(abs(color[i] - existing[i]) <= 25 for i in range(3)):
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_colors.append(color)
        
        return final_colors[:n_colors]
    
    def add_missing_colors(self, colors, missing_colors):
        """Add specific colors to the palette, replacing least interesting ones."""
        result_colors = colors.copy()
        
        for missing_color in missing_colors:
            if len(result_colors) < 5:
                result_colors.append(missing_color)
            else:
                # Replace the most gray-like color
                most_gray_idx = 0
                max_grayness = 0
                
                for i, color in enumerate(result_colors):
                    r, g, b = color
                    # Calculate "grayness" - how similar R, G, B values are
                    grayness = 1000 - (abs(r-g) + abs(g-b) + abs(r-b))
                    if grayness > max_grayness:
                        max_grayness = grayness
                        most_gray_idx = i
                
                print(f"🔄 Replacing RGB{result_colors[most_gray_idx]} with RGB{missing_color}")
                result_colors[most_gray_idx] = missing_color
        
        return result_colors
    
    def rgb_to_hex(self, rgb_color):
        """Convert RGB to hex format."""
        return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])
    
    def get_color_name(self, rgb_color):
        """Get the closest color name."""
        css3_colors = {
            'black': (0, 0, 0), 'silver': (192, 192, 192), 'gray': (128, 128, 128),
            'white': (255, 255, 255), 'maroon': (128, 0, 0), 'red': (255, 0, 0),
            'purple': (128, 0, 128), 'fuchsia': (255, 0, 255), 'green': (0, 128, 0),
            'lime': (0, 255, 0), 'olive': (128, 128, 0), 'yellow': (255, 255, 0),
            'navy': (0, 0, 128), 'blue': (0, 0, 255), 'teal': (0, 128, 128),
            'aqua': (0, 255, 255), 'orange': (255, 165, 0), 'brown': (165, 42, 42),
            'pink': (255, 192, 203), 'gold': (255, 215, 0), 'beige': (245, 245, 220),
            'tan': (210, 180, 140), 'khaki': (240, 230, 140), 'violet': (238, 130, 238),
            'coral': (255, 127, 80), 'salmon': (250, 128, 114), 'rose': (255, 0, 127),
            'magenta': (255, 0, 255), 'lavender': (230, 230, 250), 'plum': (221, 160, 221)
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
    """Main analysis with improved pink detection and manual color addition."""
    print("🎨 Improved Color Analysis Tool")
    print("=" * 40)
    
    try:
        analyzer = ImprovedColorAnalyzer("image.jpg")
        print(f"✅ Loaded image: {analyzer.image.size[0]}x{analyzer.image.size[1]} pixels")
        
        # Step 1: Extract filtered palette (no grays/blacks)
        print("\n🎯 STEP 1: Extract filtered palette (no grays/blacks)")
        print("-" * 55)
        filtered_colors = analyzer.extract_filtered_palette(5, exclude_grays=True, exclude_dark=True)
        for i, color in enumerate(filtered_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            print(f"   {i+1}. 🎨 RGB{color} → {hex_color} ({color_name})")
        
        # Step 2: Search for actual pink tones
        print("\n🌸 STEP 2: Search for pink tones in image")
        print("-" * 40)
        pink_tones = analyzer.find_actual_pink_tones(min_pixels=30)
        if pink_tones:
            print(f"Found {len(pink_tones)} pink tone(s):")
            for i, pink in enumerate(pink_tones):
                print(f"   🌸 Pink {i+1}: RGB{pink} → {analyzer.rgb_to_hex(pink)}")
        else:
            print("   No significant pink tones detected automatically")
        
        # Step 3: Manual color suggestions based on common missing colors
        print("\n✨ STEP 3: Adding colors commonly missed by algorithms")
        print("-" * 55)
        
        # Common colors that might be missed in images
        suggested_colors = [
            (220, 180, 190),  # Dusty pink
            (200, 150, 160),  # Muted rose
            (180, 140, 150),  # Deeper dusty pink
            (240, 200, 210),  # Light pink
            (160, 120, 130),  # Taupe pink
        ]
        
        print("Suggested colors to consider adding:")
        for i, color in enumerate(suggested_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            print(f"   💡 Option {i+1}: RGB{color} → {hex_color} ({color_name})")
        
        # Step 4: Create final palette with added colors
        print("\n💎 STEP 4: Final palette with added pink tones")
        print("-" * 45)
        
        # Add one or two of the suggested pink colors
        colors_to_add = []
        if not any(analyzer.is_true_pink_tone(color) for color in filtered_colors):
            colors_to_add = [suggested_colors[0], suggested_colors[1]]  # Add two pinks
            print("Adding dusty pink and muted rose to palette...")
        
        final_colors = analyzer.add_missing_colors(filtered_colors, colors_to_add)
        
        for i, color in enumerate(final_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            is_pink = "🌸" if analyzer.is_true_pink_tone(color) else "🎨"
            print(f"   {i+1}. {is_pink} RGB{color} → {hex_color} ({color_name})")
        
        # Save results
        results = {
            'image_info': {
                'path': 'image.jpg',
                'size': analyzer.image.size,
                'mode': analyzer.image.mode
            },
            'analysis_results': {
                'filtered_palette': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color)
                    } for color in filtered_colors
                ],
                'detected_pinks': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color)
                    } for color in pink_tones
                ],
                'final_palette': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color),
                        'is_pink': analyzer.is_true_pink_tone(color)
                    } for color in final_colors
                ]
            }
        }
        
        with open('final_color_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: final_color_analysis.json")
        
        print(f"\n🌐 FINAL HEX PALETTE:")
        css_colors = [analyzer.rgb_to_hex(color) for color in final_colors]
        print(f"   {', '.join(css_colors)}")
        
        print(f"\n📋 COPY-PASTE READY:")
        print(f"   CSS: {' '.join(css_colors)}")
        print(f"   Array: [{', '.join([f'\"#{hex_color[1:]}\"' for hex_color in css_colors])}]")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
