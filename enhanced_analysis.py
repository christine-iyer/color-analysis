#!/usr/bin/env python3
"""
Enhanced color analysis with filtering, pink detection, and color substitution.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import colorthief
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class EnhancedColorAnalyzer:
    def __init__(self, image_path):
        """Initialize the color analyzer with an image path."""
        self.image_path = image_path
        self.image = Image.open(image_path)
        
    def is_grayscale_color(self, rgb_color, tolerance=30):
        """Check if a color is grayscale (R, G, B values are similar)."""
        r, g, b = rgb_color
        return (abs(r - g) <= tolerance and 
                abs(g - b) <= tolerance and 
                abs(r - b) <= tolerance)
    
    def is_dark_color(self, rgb_color, threshold=50):
        """Check if a color is too dark."""
        return sum(rgb_color) / 3 < threshold
    
    def is_pink_tone(self, rgb_color):
        """Detect if a color has pink characteristics."""
        r, g, b = rgb_color
        
        # Pink detection criteria:
        # 1. Red component should be higher than blue and green
        # 2. Red should be reasonably strong
        # 3. Check various pink ranges
        
        # Classic pink: high red, moderate green, moderate-high blue
        if r > g and r > b and r > 100:
            if (r - g) > 20 and (r - b) > 10:
                return True
        
        # Dusty/muted pink: similar to above but lower intensity
        if r >= g and r >= b and r > 80:
            if (r - g) >= 10 and (r - b) >= 5:
                # Check if it's not too gray
                if not self.is_grayscale_color(rgb_color, tolerance=20):
                    return True
        
        # Rose tones: red dominant with some blue
        if r > 120 and b > 80 and r > g:
            if (r - g) > 30:
                return True
                
        return False
    
    def find_pink_tones(self, min_pixels=100):
        """Specifically search for pink tones in the image."""
        if self.image.mode != 'RGB':
            rgb_image = self.image.convert('RGB')
        else:
            rgb_image = self.image
            
        pixels = list(rgb_image.getdata())
        pink_colors = []
        
        # Group similar pink colors
        pink_groups = {}
        for pixel in pixels:
            if self.is_pink_tone(pixel):
                # Group similar pinks together
                found_group = False
                for group_color in pink_groups:
                    if all(abs(pixel[i] - group_color[i]) <= 15 for i in range(3)):
                        pink_groups[group_color] += 1
                        found_group = True
                        break
                if not found_group:
                    pink_groups[pixel] = 1
        
        # Filter by minimum pixel count and return most common pinks
        significant_pinks = [(color, count) for color, count in pink_groups.items() 
                           if count >= min_pixels]
        significant_pinks.sort(key=lambda x: x[1], reverse=True)
        
        return [color for color, count in significant_pinks[:3]]  # Top 3 pink tones
    
    def extract_filtered_palette(self, n_colors=5, exclude_grays=True, 
                                exclude_dark=True, include_pinks=True, resize_width=150):
        """
        Extract color palette with filtering options.
        
        Args:
            n_colors: Number of colors to extract
            exclude_grays: Remove grayscale colors
            exclude_dark: Remove very dark colors  
            include_pinks: Force include pink tones if found
            resize_width: Width for image resizing
        """
        # Resize image for faster processing
        aspect_ratio = self.image.height / self.image.width
        new_height = int(resize_width * aspect_ratio)
        resized_image = self.image.resize((resize_width, new_height))
        
        # Convert to numpy array and reshape
        data = np.array(resized_image)
        data = data.reshape((-1, 3))
        data = data.astype(np.float64)
        
        # Apply K-means clustering with more clusters initially
        initial_clusters = min(n_colors * 3, 15)  # Extract more colors initially
        kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
        kmeans.fit(data)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        colors = colors.round(0).astype(int)
        candidate_colors = [tuple(int(c) for c in color) for color in colors]
        
        # Filter colors based on criteria
        filtered_colors = []
        for color in candidate_colors:
            # Skip if it's grayscale and we're excluding grays
            if exclude_grays and self.is_grayscale_color(color):
                continue
                
            # Skip if it's too dark and we're excluding dark colors
            if exclude_dark and self.is_dark_color(color):
                continue
                
            filtered_colors.append(color)
        
        # Find pink tones in original image
        pink_tones = []
        if include_pinks:
            pink_tones = self.find_pink_tones()
            print(f"🌸 Found {len(pink_tones)} significant pink tones")
            for i, pink in enumerate(pink_tones):
                print(f"   Pink {i+1}: RGB{pink} → {self.rgb_to_hex(pink)}")
        
        # Combine filtered colors with pink tones
        final_colors = []
        
        # Add pink tones first (priority)
        for pink in pink_tones:
            if len(final_colors) < n_colors:
                final_colors.append(pink)
        
        # Add other filtered colors
        for color in filtered_colors:
            if len(final_colors) >= n_colors:
                break
            # Avoid duplicates (colors too similar to existing ones)
            is_duplicate = False
            for existing in final_colors:
                if all(abs(color[i] - existing[i]) <= 30 for i in range(3)):
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_colors.append(color)
        
        # If we still don't have enough colors, add some of the best candidates
        if len(final_colors) < n_colors:
            remaining_needed = n_colors - len(final_colors)
            for color in candidate_colors:
                if len(final_colors) >= n_colors:
                    break
                is_duplicate = False
                for existing in final_colors:
                    if all(abs(color[i] - existing[i]) <= 20 for i in range(3)):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    final_colors.append(color)
        
        return final_colors[:n_colors]
    
    def substitute_color(self, colors, old_color_approx, new_color):
        """
        Replace a color in the palette with a new color.
        
        Args:
            colors: List of RGB color tuples
            old_color_approx: RGB tuple of color to replace (approximate match)
            new_color: RGB tuple of replacement color
        """
        substituted_colors = []
        replaced = False
        
        for color in colors:
            # Find the closest match to old_color_approx
            distance = sum((color[i] - old_color_approx[i])**2 for i in range(3))
            if distance < 5000 and not replaced:  # Threshold for "close enough"
                substituted_colors.append(new_color)
                replaced = True
                print(f"🔄 Replaced RGB{color} with RGB{new_color}")
            else:
                substituted_colors.append(color)
        
        if not replaced:
            print(f"⚠️  Could not find close match for RGB{old_color_approx}")
            
        return substituted_colors
    
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
            'tan': (210, 180, 140), 'khaki': (240, 230, 140), 'violet': (238, 130, 238),
            'coral': (255, 127, 80), 'salmon': (250, 128, 114), 'rose': (255, 0, 127),
            'magenta': (255, 0, 255), 'lavender': (230, 230, 250)
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

    def create_palette_visualization(self, colors, title="Color Palette", save_path=None):
        """
        Create and save a visual representation of the color palette.
        
        Args:
            colors: List of RGB color tuples
            title: Title for the palette
            save_path: Path to save the image (if None, uses title)
        """
        if save_path is None:
            save_path = f"{title.lower().replace(' ', '_')}_palette.png"
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top subplot: Color swatches
        for i, color in enumerate(colors):
            # Normalize RGB values for matplotlib
            norm_color = [c/255.0 for c in color]
            
            # Create rectangle for each color
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, 
                                   edgecolor='white', facecolor=norm_color)
            ax1.add_patch(rect)
            
            # Add color information text
            hex_color = self.rgb_to_hex(color)
            rgb_text = f'RGB{color}'
            color_name = self.get_color_name(color)
            
            # Determine text color based on brightness
            brightness = sum(color) / 3
            text_color = 'white' if brightness < 128 else 'black'
            
            # Add text with color info
            ax1.text(i + 0.5, 0.5, f'{hex_color}\n{rgb_text}\n{color_name}', 
                    ha='center', va='center', fontsize=9, color=text_color, 
                    weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='black', alpha=0.1))
        
        # Configure top subplot
        ax1.set_xlim(0, len(colors))
        ax1.set_ylim(0, 1)
        ax1.set_title(f'{title} - {len(colors)} Colors', fontsize=14, weight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_aspect('equal')
        
        # Remove top subplot borders
        for spine in ax1.spines.values():
            spine.set_visible(False)
        
        # Bottom subplot: Original image
        ax2.imshow(self.image)
        ax2.set_title('Original Image', fontsize=12, weight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close to avoid display issues
        
        return save_path

    def create_comparison_visualization(self, palettes_dict, save_path="palette_comparison.png"):
        """
        Create a comparison visualization of multiple palettes.
        
        Args:
            palettes_dict: Dictionary with palette names as keys and color lists as values
            save_path: Path to save the comparison image
        """
        n_palettes = len(palettes_dict)
        fig, axes = plt.subplots(n_palettes + 1, 1, figsize=(14, 2 * (n_palettes + 1)))
        
        # Make axes a list if there's only one subplot
        if n_palettes == 0:
            return
        if n_palettes == 1:
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
        
        # Original image at the top
        axes[0].imshow(self.image)
        axes[0].set_title('Original Image', fontsize=12, weight='bold')
        axes[0].axis('off')
        
        # Create palette visualizations
        for idx, (palette_name, colors) in enumerate(palettes_dict.items()):
            ax = axes[idx + 1]
            
            for i, color in enumerate(colors):
                # Normalize RGB values
                norm_color = [c/255.0 for c in color]
                
                # Create rectangle
                rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, 
                                       edgecolor='white', facecolor=norm_color)
                ax.add_patch(rect)
                
                # Add hex color text
                hex_color = self.rgb_to_hex(color)
                brightness = sum(color) / 3
                text_color = 'white' if brightness < 128 else 'black'
                
                ax.text(i + 0.5, 0.5, hex_color, ha='center', va='center', 
                       fontsize=10, color=text_color, weight='bold')
            
            # Configure subplot
            ax.set_xlim(0, len(colors))
            ax.set_ylim(0, 1)
            ax.set_title(f'{palette_name}', fontsize=11, weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove borders
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path

def main():
    """Main analysis function with filtering and substitution."""
    print("🎨 Enhanced Color Analysis Tool")
    print("=" * 40)
    
    try:
        analyzer = EnhancedColorAnalyzer("image.jpg")
        print(f"✅ Loaded image: {analyzer.image.size[0]}x{analyzer.image.size[1]} pixels")
        
        # Original analysis (for comparison)
        print("\n📊 ORIGINAL ANALYSIS (K-means, no filtering):")
        print("-" * 45)
        original_colors = analyzer.extract_filtered_palette(5, exclude_grays=False, 
                                                           exclude_dark=False, 
                                                           include_pinks=False)
        for i, color in enumerate(original_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            is_gray = "🔘" if analyzer.is_grayscale_color(color) else "🎨"
            print(f"   {i+1}. {is_gray} RGB{color} → {hex_color} ({color_name})")
        
        # Enhanced analysis with filtering
        print("\n✨ ENHANCED ANALYSIS (Filtered + Pink Detection):")
        print("-" * 50)
        enhanced_colors = analyzer.extract_filtered_palette(5, exclude_grays=True, 
                                                           exclude_dark=True, 
                                                           include_pinks=True)
        for i, color in enumerate(enhanced_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            is_pink = "🌸" if analyzer.is_pink_tone(color) else "🎨"
            print(f"   {i+1}. {is_pink} RGB{color} → {hex_color} ({color_name})")
        
        # Example color substitution
        print("\n🔄 COLOR SUBSTITUTION EXAMPLE:")
        print("-" * 35)
        print("Let's replace any remaining gray with a nice pink...")
        
        # Find the most gray-like color in enhanced palette
        substituted_colors = enhanced_colors.copy()
        for color in enhanced_colors:
            if analyzer.is_grayscale_color(color, tolerance=40):
                # Replace with a nice dusty pink
                dusty_pink = (200, 150, 160)  # Custom dusty pink
                substituted_colors = analyzer.substitute_color(
                    substituted_colors, color, dusty_pink)
                break
        
        print("\n💎 FINAL SUBSTITUTED PALETTE:")
        print("-" * 30)
        for i, color in enumerate(substituted_colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            is_pink = "🌸" if analyzer.is_pink_tone(color) else "🎨"
            print(f"   {i+1}. {is_pink} RGB{color} → {hex_color} ({color_name})")
        
        # Save results
        results = {
            'image_info': {
                'path': 'image.jpg',
                'size': analyzer.image.size,
                'mode': analyzer.image.mode
            },
            'analysis_results': {
                'original_palette': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color),
                        'is_grayscale': analyzer.is_grayscale_color(color)
                    } for color in original_colors
                ],
                'enhanced_palette': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color),
                        'is_pink': analyzer.is_pink_tone(color)
                    } for color in enhanced_colors
                ],
                'final_palette': [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color),
                        'is_pink': analyzer.is_pink_tone(color)
                    } for color in substituted_colors
                ]
            }
        }
        
        with open('enhanced_color_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Enhanced results saved to: enhanced_color_analysis.json")
        
        # Create visualizations
        print(f"\n🖼️  Creating palette visualizations...")
        
        try:
            # Individual palette visualizations
            original_viz = analyzer.create_palette_visualization(
                original_colors, "Original Analysis (K-means)")
            print(f"   📁 Original palette: {original_viz}")
            
            enhanced_viz = analyzer.create_palette_visualization(
                enhanced_colors, "Enhanced Analysis (Filtered + Pink Detection)")
            print(f"   📁 Enhanced palette: {enhanced_viz}")
            
            final_viz = analyzer.create_palette_visualization(
                substituted_colors, "Final Substituted Palette")
            print(f"   📁 Final palette: {final_viz}")
            
            # Comparison visualization
            palettes_comparison = {
                "Original (with grays/darks)": original_colors,
                "Enhanced (filtered)": enhanced_colors,
                "Final (substituted)": substituted_colors
            }
            
            comparison_viz = analyzer.create_comparison_visualization(palettes_comparison)
            print(f"   📁 Comparison view: {comparison_viz}")
            
        except Exception as viz_error:
            print(f"   ⚠️  Visualization error: {viz_error}")
            print(f"   📄 Color data is still available in the JSON file")
        
        # CSS/HTML color palette output
        print(f"\n🌐 CSS HEX COLORS (Final Palette):")
        css_colors = [analyzer.rgb_to_hex(color) for color in substituted_colors]
        print(f"   {', '.join(css_colors)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
