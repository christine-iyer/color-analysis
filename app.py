#!/usr/bin/env python3
"""
Web frontend for color analysis tool.
Flask application with image upload and visualization display.
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from datetime import datetime
import cv2
import colorthief
from collections import Counter
import webcolors

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class WebColorAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        
    def is_grayscale_color(self, rgb_color, tolerance=25):
        r, g, b = rgb_color
        return (abs(r - g) <= tolerance and 
                abs(g - b) <= tolerance and 
                abs(r - b) <= tolerance)
    
    def is_dark_color(self, rgb_color, threshold=60):
        return sum(rgb_color) / 3 < threshold
    
    def is_pink_tone(self, rgb_color):
        r, g, b = [x/255.0 for x in rgb_color]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        if diff == 0:
            return False
            
        if max_val == r:
            hue = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            hue = (60 * ((b - r) / diff) + 120) % 360
        else:
            hue = (60 * ((r - g) / diff) + 240) % 360
        
        is_pink_hue = (300 <= hue <= 360) or (0 <= hue <= 30)
        has_red_dominance = rgb_color[0] >= rgb_color[1] and rgb_color[0] >= rgb_color[2]
        sufficient_brightness = max_val > 0.3
        
        return is_pink_hue and has_red_dominance and sufficient_brightness
    
    def extract_palette_basic(self, n_colors=5):
        """Basic K-means extraction (original method)"""
        resized = self.image.resize((150, int(150 * self.image.height / self.image.width)))
        data = np.array(resized).reshape((-1, 3)).astype(np.float64)
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(data)
        
        colors = kmeans.cluster_centers_.round(0).astype(int)
        return [tuple(int(c) for c in color) for color in colors]
    
    def extract_palette_gaussian_mixture(self, n_colors=5):
        """Gaussian Mixture Model for color extraction"""
        resized = self.image.resize((150, int(150 * self.image.height / self.image.width)))
        data = np.array(resized).reshape((-1, 3)).astype(np.float64)
        
        # Normalize data for better GMM performance
        data = data / 255.0
        
        gmm = GaussianMixture(n_components=n_colors, random_state=42, covariance_type='full')
        gmm.fit(data)
        
        # Get cluster centers (means) and convert back to RGB
        colors = (gmm.means_ * 255).round(0).astype(int)
        return [tuple(int(c) for c in color) for color in colors]
    
    def extract_palette_dbscan(self, n_colors=5):
        """DBSCAN clustering for automatic color grouping"""
        resized = self.image.resize((100, int(100 * self.image.height / self.image.width)))
        data = np.array(resized).reshape((-1, 3)).astype(np.float64)
        
        # DBSCAN with adaptive epsilon
        dbscan = DBSCAN(eps=15, min_samples=50)
        clusters = dbscan.fit_predict(data)
        
        # Get cluster centers
        unique_clusters = np.unique(clusters)
        colors = []
        
        for cluster in unique_clusters:
            if cluster != -1:  # Ignore noise points
                cluster_points = data[clusters == cluster]
                center = np.mean(cluster_points, axis=0).round(0).astype(int)
                colors.append(tuple(int(c) for c in center))
        
        # If we don't have enough clusters, fallback to K-means
        if len(colors) < n_colors:
            return self.extract_palette_basic(n_colors)
        
        return colors[:n_colors]
    
    def extract_palette_colorthief(self, n_colors=5):
        """ColorThief library for perceptually-based extraction"""
        try:
            # Save image temporarily for ColorThief
            temp_path = 'temp_image.jpg'
            self.image.save(temp_path, 'JPEG')
            
            color_thief = colorthief.ColorThief(temp_path)
            
            if n_colors == 1:
                dominant = color_thief.get_color(quality=1)
                colors = [dominant]
            else:
                colors = color_thief.get_palette(color_count=n_colors, quality=1)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return colors
        except Exception:
            # Fallback to K-means if ColorThief fails
            return self.extract_palette_basic(n_colors)
    
    def extract_palette_minibatch_kmeans(self, n_colors=5):
        """MiniBatch K-means for faster processing of large images"""
        # Use larger image for MiniBatch (it's more efficient)
        resized = self.image.resize((200, int(200 * self.image.height / self.image.width)))
        data = np.array(resized).reshape((-1, 3)).astype(np.float64)
        
        mbk = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=1000)
        mbk.fit(data)
        
        colors = mbk.cluster_centers_.round(0).astype(int)
        return [tuple(int(c) for c in color) for color in colors]
    
    def extract_palette_quantization(self, n_colors=5):
        """OpenCV color quantization using K-means"""
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        
        # Resize for processing
        height, width = cv_image.shape[:2]
        new_width = 150
        new_height = int(150 * height / width)
        resized = cv2.resize(cv_image, (new_width, new_height))
        
        # Reshape to a 2D array of pixels
        data = resized.reshape((-1, 3)).astype(np.float32)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert BGR back to RGB and to integers
        centers_rgb = centers[:, ::-1].round(0).astype(int)
        return [tuple(int(c) for c in color) for color in centers_rgb]
    
    def extract_palette_histogram_peaks(self, n_colors=5):
        """Extract colors based on histogram peaks in HSV space"""
        # Convert to HSV for better color separation
        hsv_image = self.image.convert('HSV')
        hsv_array = np.array(hsv_image)
        
        # Resize for processing
        resized = cv2.resize(hsv_array, (150, int(150 * self.image.height / self.image.width)))
        
        # Calculate histogram for hue channel
        hue_hist = cv2.calcHist([resized], [0], None, [180], [0, 180])
        
        # Find peaks in hue histogram
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hue_hist.flatten(), height=np.max(hue_hist) * 0.1)
        except ImportError:
            # Fallback: simple peak detection
            peaks = []
            hist_smooth = hue_hist.flatten()
            for i in range(1, len(hist_smooth) - 1):
                if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                    if hist_smooth[i] > np.max(hist_smooth) * 0.1:
                        peaks.append(i)
        
        # Extract colors around peak hues
        colors = []
        for peak in peaks[:n_colors]:
            # Find pixels with similar hue
            hue_mask = np.abs(resized[:, :, 0] - peak) < 10
            if np.any(hue_mask):
                # Get average RGB color for this hue region
                rgb_resized = cv2.resize(np.array(self.image), (150, int(150 * self.image.height / self.image.width)))
                masked_pixels = rgb_resized[hue_mask]
                if len(masked_pixels) > 0:
                    avg_color = np.mean(masked_pixels, axis=0).round(0).astype(int)
                    colors.append(tuple(int(c) for c in avg_color))
        
        # Fill remaining with K-means if needed
        while len(colors) < n_colors:
            remaining = self.extract_palette_basic(n_colors - len(colors))
            colors.extend(remaining)
        
        return colors[:n_colors]
    
    def extract_palette_filtered(self, n_colors=5):
        """Filtered extraction (excludes grays and darks)"""
        colors = self.extract_palette_basic(n_colors * 2)  # Get more colors initially
        
        filtered = []
        for color in colors:
            if not self.is_grayscale_color(color) and not self.is_dark_color(color):
                filtered.append(color)
        
        # Remove duplicates
        final = []
        for color in filtered:
            is_duplicate = False
            for existing in final:
                if all(abs(color[i] - existing[i]) <= 25 for i in range(3)):
                    is_duplicate = True
                    break
            if not is_duplicate:
                final.append(color)
        
        return final[:n_colors]
    
    def extract_palette_pink_enhanced(self, n_colors=5):
        """Pink-enhanced extraction with manual additions"""
        filtered = self.extract_palette_filtered(n_colors - 2)
        
        # Add some pink tones
        pink_additions = [
            (220, 180, 190),  # Soft pink
            (200, 150, 160),  # Rose
        ]
        
        # Add pinks if there aren't any detected
        has_pink = any(self.is_pink_tone(color) for color in filtered)
        if not has_pink:
            for pink in pink_additions:
                if len(filtered) < n_colors:
                    filtered.append(pink)
        
        return filtered[:n_colors]
    
    def rgb_to_hex(self, rgb_color):
        return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])
    
    def get_color_name(self, rgb_color):
        css3_colors = {
            'black': (0, 0, 0), 'gray': (128, 128, 128), 'silver': (192, 192, 192),
            'white': (255, 255, 255), 'red': (255, 0, 0), 'lime': (0, 255, 0),
            'blue': (0, 0, 255), 'yellow': (255, 255, 0), 'cyan': (0, 255, 255),
            'magenta': (255, 0, 255), 'maroon': (128, 0, 0), 'olive': (128, 128, 0),
            'green': (0, 128, 0), 'purple': (128, 0, 128), 'teal': (0, 128, 128),
            'navy': (0, 0, 128), 'orange': (255, 165, 0), 'pink': (255, 192, 203),
            'gold': (255, 215, 0), 'brown': (165, 42, 42), 'tan': (210, 180, 140),
            'coral': (255, 127, 80), 'salmon': (250, 128, 114)
        }
        
        min_distance = float('inf')
        closest_name = 'custom'
        
        for name, (r_c, g_c, b_c) in css3_colors.items():
            distance = sum((rgb_color[i] - (r_c, g_c, b_c)[i])**2 for i in range(3))**0.5
            if distance < min_distance:
                min_distance = distance
                closest_name = name
                
        return closest_name
    
    def create_web_visualization(self, palettes_dict, filename_base):
        """Create visualization optimized for web display"""
        n_palettes = len(palettes_dict)
        fig, axes = plt.subplots(n_palettes + 1, 1, figsize=(12, 2 * (n_palettes + 1)))
        
        if n_palettes == 1:
            axes = [axes]
        
        # Original image at the top
        axes[0].imshow(self.image)
        axes[0].set_title('Uploaded Image', fontsize=14, weight='bold')
        axes[0].axis('off')
        
        # Palette visualizations
        for idx, (palette_name, colors) in enumerate(palettes_dict.items()):
            ax = axes[idx + 1]
            
            for i, color in enumerate(colors):
                norm_color = [c/255.0 for c in color]
                rect = patches.Rectangle((i, 0), 1, 1, linewidth=2, 
                                       edgecolor='white', facecolor=norm_color)
                ax.add_patch(rect)
                
                hex_color = self.rgb_to_hex(color)
                brightness = sum(color) / 3
                text_color = 'white' if brightness < 128 else 'black'
                
                ax.text(i + 0.5, 0.5, hex_color, ha='center', va='center', 
                       fontsize=11, color=text_color, weight='bold')
            
            ax.set_xlim(0, len(colors))
            ax.set_ylim(0, 1)
            ax.set_title(f'{palette_name}', fontsize=12, weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        
        # Save to static folder for web serving
        output_path = f'static/results/{filename_base}_comparison.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Analyze the image
            analyzer = WebColorAnalyzer(filepath)
            
            # Extract palettes using all available methods
            methods = {
                'K-means Clustering': analyzer.extract_palette_basic,
                'Gaussian Mixture Model': analyzer.extract_palette_gaussian_mixture,
                'ColorThief (Perceptual)': analyzer.extract_palette_colorthief,
                'MiniBatch K-means': analyzer.extract_palette_minibatch_kmeans,
                'OpenCV Quantization': analyzer.extract_palette_quantization,
                'DBSCAN Clustering': analyzer.extract_palette_dbscan,
                'Filtered (No Grays)': analyzer.extract_palette_filtered,
                'Pink Enhanced': analyzer.extract_palette_pink_enhanced
            }
            
            palettes = {}
            for name, method in methods.items():
                try:
                    palettes[name] = method(5)
                except Exception as e:
                    print(f"Error with {name}: {e}")
                    # Fallback to basic method
                    palettes[name] = analyzer.extract_palette_basic(5)
            
            # Create visualization with all methods
            viz_path = analyzer.create_web_visualization(palettes, timestamp)
            
            # Prepare results
            results = {
                'success': True,
                'image_info': {
                    'filename': filename,
                    'size': f"{analyzer.image.width}x{analyzer.image.height}",
                    'mode': analyzer.image.mode
                },
                'visualization': viz_path.replace('static/', ''),
                'palettes': {},
                'method_descriptions': {
                    'K-means Clustering': 'Traditional clustering algorithm, groups similar colors',
                    'Gaussian Mixture Model': 'Probabilistic model, captures color distributions',
                    'ColorThief (Perceptual)': 'Library optimized for human color perception',
                    'MiniBatch K-means': 'Faster K-means variant for large images',
                    'OpenCV Quantization': 'Computer vision approach with color reduction',
                    'DBSCAN Clustering': 'Density-based clustering, finds natural color groups',
                    'Filtered (No Grays)': 'K-means with grayscale and dark color filtering',
                    'Pink Enhanced': 'Filtered approach with manual pink tone additions'
                }
            }
            
            # Add palette data
            for name, colors in palettes.items():
                results['palettes'][name] = [
                    {
                        'rgb': color,
                        'hex': analyzer.rgb_to_hex(color),
                        'name': analyzer.get_color_name(color),
                        'is_pink': analyzer.is_pink_tone(color),
                        'is_gray': analyzer.is_grayscale_color(color)
                    } for color in colors
                ]
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(results)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    print("🎨 Color Analysis Web App")
    print("=" * 30)
    print("🌐 Starting server...")
    print("📱 Open your browser to: http://localhost:5001")
    print("⏹️  Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5001)
