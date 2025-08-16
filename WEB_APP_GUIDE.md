# 🎨 Color Analysis Web Application

## 🚀 **Your Frontend is Now Live!**

The web application is running at: **http://localhost:5001**

## ✨ **Features:**

### 🖱️ **Easy Upload Interface**
- **Drag & drop** images directly onto the upload area
- **Click to browse** and select image files
- **Support for multiple formats**: JPG, PNG, GIF, WebP, BMP, TIFF
- **File size limit**: 16MB maximum

### 🎯 **Three Analysis Methods**
1. **Original K-means**: Basic color extraction (includes grays/darks)
2. **Filtered**: Excludes grays and dark colors for cleaner palettes
3. **Pink Enhanced**: Specifically adds pink tones that might be missed

### 🖼️ **Visual Results**
- **Side-by-side comparison** of all three methods
- **Original image** displayed with extracted palettes
- **Color swatches** with hex codes and names
- **High-quality visualizations** saved as PNG files

### 📋 **Copy-Ready Formats**
- **CSS color strings**: `#ff6b6b #4ecdc4 #45b7d1`
- **Array format**: `["#ff6b6b", "#4ecdc4", "#45b7d1"]`
- **Click to select** for easy copying

### 🏷️ **Smart Color Detection**
- **Pink tone indicators**: 🌸 badge for detected pink colors
- **Gray indicators**: 🔘 badge for grayscale colors
- **Color names**: Closest CSS color name for each shade

## 🎮 **How to Use:**

1. **Open your browser** to http://localhost:5001
2. **Upload an image** by dragging/dropping or clicking to browse
3. **Click "Analyze Colors"** to process the image
4. **View the results** with visual comparisons and detailed color information
5. **Copy the hex codes** you like for use in your projects

## 🛠️ **Technical Details:**

- **Frontend**: HTML5, CSS3, JavaScript (no external dependencies)
- **Backend**: Flask Python web framework
- **Analysis**: scikit-learn K-means clustering + custom filtering
- **Visualization**: matplotlib with web-optimized output
- **File handling**: Secure upload with automatic cleanup

## 📱 **Mobile Friendly:**
The interface is responsive and works well on mobile devices and tablets.

## 🔄 **Process Multiple Images:**
After each analysis, click "Analyze Another Image" to reset and upload a new file.

## ⚠️ **Note:**
This is a development server. Files are automatically deleted after processing for security.

---

**Enjoy extracting beautiful color palettes from your images! 🎨**
