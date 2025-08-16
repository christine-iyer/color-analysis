#!/usr/bin/env python3
"""
Color analysis that saves results to files instead of displaying.
"""

from color_analyzer import ColorAnalyzer
import matplotlib.pyplot as plt

def save_analysis():
    """Save color analysis results to files."""
    print("🎨 Color Analysis Tool - File Output Mode")
    print("=" * 45)
    
    # Initialize analyzer
    analyzer = ColorAnalyzer("image.jpg")
    
    # Extract colors using all methods
    print("Extracting colors using different methods...\n")
    
    methods = {
        'K-means Clustering': analyzer.extract_palette_kmeans,
        'ColorThief': analyzer.extract_palette_colorthief,
        'Most Common Colors': analyzer.extract_most_common_colors
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        try:
            print(f"Running {method_name}...")
            colors = method_func(5)
            results[method_name] = colors
            
            print(f"  Found colors: {colors}")
            
            # Create and save visualization
            fig = analyzer.display_palette(colors, method_name, show_names=True)
            filename = f"palette_{method_name.lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved visualization: {filename}")
            
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            results[method_name] = None
    
    # Save detailed analysis to text file
    with open('detailed_color_analysis.txt', 'w') as f:
        f.write("DETAILED COLOR ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Image: image.jpg\n")
        f.write(f"Number of colors extracted: 5\n\n")
        
        for method_name, colors in results.items():
            if colors:
                f.write(f"{method_name.upper()}:\n")
                f.write("-" * 30 + "\n")
                
                for i, color in enumerate(colors):
                    hex_color = analyzer.rgb_to_hex(color)
                    try:
                        color_name = analyzer.get_color_name(color)
                        f.write(f"Color {i+1}: RGB{tuple(int(c) for c in color)} | {hex_color} | {color_name}\n")
                    except:
                        f.write(f"Color {i+1}: RGB{tuple(int(c) for c in color)} | {hex_color}\n")
                f.write("\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Detailed results saved to: detailed_color_analysis.txt")
    print(f"🖼️  Visual palettes saved as PNG files")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    if 'K-means Clustering' in results and results['K-means Clustering']:
        colors = results['K-means Clustering']
        print(f"Dominant colors (K-means method):")
        for i, color in enumerate(colors):
            hex_color = analyzer.rgb_to_hex(color)
            color_name = analyzer.get_color_name(color)
            print(f"  {i+1}. {hex_color} ({color_name})")

if __name__ == "__main__":
    save_analysis()
