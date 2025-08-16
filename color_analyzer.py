import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import colorthief
import webcolors
import cv2
import os


class ColorAnalyzer:
    def __init__(self, image_path):
        """Initialize the color analyzer with an image path."""
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.image_array = np.array(self.image)
        
    def extract_palette_kmeans(self, n_colors=5, resize_width=150):
        """
        Extract color palette using K-means clustering.
        
        Args:
            n_colors (int): Number of colors to extract
            resize_width (int): Width to resize image for faster processing
            
        Returns:
            list: RGB values of dominant colors
        """
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
        
        return [tuple(color) for color in colors]
    
    def extract_palette_colorthief(self, n_colors=5):
        """
        Extract color palette using ColorThief library.
        
        Args:
            n_colors (int): Number of colors to extract
            
        Returns:
            list: RGB values of dominant colors
        """
        color_thief = colorthief.ColorThief(self.image_path)
        
        # Get the dominant color
        dominant_color = color_thief.get_color(quality=1)
        
        # Get the color palette
        if n_colors > 1:
            palette = color_thief.get_palette(color_count=n_colors, quality=1)
            return palette
        else:
            return [dominant_color]
    
    def extract_most_common_colors(self, n_colors=5, tolerance=10):
        """
        Extract most common colors by frequency.
        
        Args:
            n_colors (int): Number of colors to extract
            tolerance (int): Color similarity tolerance for grouping
            
        Returns:
            list: RGB values of most common colors
        """
        # Convert image to RGB if necessary
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
    
    def get_color_name(self, rgb_color):
        """
        Get the closest color name for an RGB value.
        
        Args:
            rgb_color (tuple): RGB color tuple
            
        Returns:
            str: Color name
        """
        # Convert numpy integers to regular integers
        rgb_color = tuple(int(c) for c in rgb_color)
        
        try:
            closest_name = webcolors.rgb_to_name(rgb_color)
        except ValueError:
            # Find the closest color using CSS3 color names
            min_colours = {}
            # Get CSS3 colors
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
            
            for name, (r_c, g_c, b_c) in css3_colors.items():
                rd = (r_c - rgb_color[0]) ** 2
                gd = (g_c - rgb_color[1]) ** 2
                bd = (b_c - rgb_color[2]) ** 2
                min_colours[(rd + gd + bd)] = name
            closest_name = min_colours[min(min_colours.keys())]
        return closest_name
    
    def rgb_to_hex(self, rgb_color):
        """Convert RGB to hex format."""
        # Convert numpy integers to regular integers
        rgb_color = tuple(int(c) for c in rgb_color)
        return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])
    
    def display_palette(self, colors, method_name="Color Palette", show_names=True):
        """
        Display the color palette as a matplotlib plot.
        
        Args:
            colors (list): List of RGB color tuples
            method_name (str): Name of the extraction method
            show_names (bool): Whether to show color names
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Display colors as rectangles
        for i, color in enumerate(colors):
            rect = plt.Rectangle((i, 0), 1, 1, facecolor=np.array(color)/255.0)
            axes[0].add_patch(rect)
            
            # Add color information
            hex_color = self.rgb_to_hex(color)
            rgb_text = f'RGB{color}'
            
            if show_names:
                color_name = self.get_color_name(color)
                text = f'{hex_color}\n{rgb_text}\n{color_name}'
            else:
                text = f'{hex_color}\n{rgb_text}'
                
            # Choose text color based on brightness
            brightness = sum(color) / 3
            text_color = 'white' if brightness < 128 else 'black'
            
            axes[0].text(i + 0.5, 0.5, text, ha='center', va='center', 
                        fontsize=8, color=text_color, weight='bold')
        
        axes[0].set_xlim(0, len(colors))
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f'{method_name} - {len(colors)} Colors')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Display original image
        axes[1].imshow(self.image)
        axes[1].set_title('Original Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_all_methods(self, n_colors=5):
        """
        Run all color extraction methods and display results.
        
        Args:
            n_colors (int): Number of colors to extract
        """
        print(f"Analyzing '{self.image_path}' for {n_colors} dominant colors...\n")
        
        # Method 1: K-means clustering
        print("1. K-means Clustering Method:")
        kmeans_colors = self.extract_palette_kmeans(n_colors)
        print(f"Colors: {kmeans_colors}")
        self.display_palette(kmeans_colors, "K-means Clustering")
        
        # Method 2: ColorThief
        print("\n2. ColorThief Method:")
        try:
            colorthief_colors = self.extract_palette_colorthief(n_colors)
            print(f"Colors: {colorthief_colors}")
            self.display_palette(colorthief_colors, "ColorThief")
        except Exception as e:
            print(f"ColorThief failed: {e}")
        
        # Method 3: Most common colors
        print("\n3. Most Common Colors Method:")
        common_colors = self.extract_most_common_colors(n_colors)
        print(f"Colors: {common_colors}")
        self.display_palette(common_colors, "Most Common Colors")
        
        return {
            'kmeans': kmeans_colors,
            'colorthief': colorthief_colors if 'colorthief_colors' in locals() else None,
            'common': common_colors
        }


def main():
    """Main function to demonstrate color analysis."""
    image_path = "image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found in current directory.")
        return
    
    # Create analyzer
    analyzer = ColorAnalyzer(image_path)
    
    # Analyze with different numbers of colors
    for n_colors in [3, 5, 8]:
        print(f"\n{'='*50}")
        print(f"ANALYSIS WITH {n_colors} COLORS")
        print(f"{'='*50}")
        results = analyzer.analyze_all_methods(n_colors)
        
        # Save results to file
        with open(f'color_analysis_{n_colors}_colors.txt', 'w') as f:
            f.write(f"Color Analysis Results for {n_colors} colors\n")
            f.write("="*50 + "\n\n")
            
            for method, colors in results.items():
                if colors:
                    f.write(f"{method.upper()} METHOD:\n")
                    for i, color in enumerate(colors):
                        hex_color = analyzer.rgb_to_hex(color)
                        color_name = analyzer.get_color_name(color)
                        f.write(f"  Color {i+1}: {color} | {hex_color} | {color_name}\n")
                    f.write("\n")


if __name__ == "__main__":
    main()
