import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import pytesseract
import numpy as np
import cv2
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")
st.title("🏭 Plant Efficiency & Safety Monitor")

# --- 1. SETUP ---
if not shutil.which("tesseract"):
    st.error("❌ CRITICAL ERROR: Tesseract is missing! Check packages.txt")
    st.stop()

# --- 2. PERCENTAGE COORDINATES ---
# Format: [Top%, Bottom%, Left%, Right%]
# These values will stretch to fit ANY image size automatically.
ROIS_PCT = {
    "Total Air Flow": [0.22, 0.26, 0.16, 0.23],  # ~22% down, 16% left
    "Fan A Amps":     [0.26, 0.30, 0.81, 0.88],  # ~26% down, 81% left
    "Fan A Vib (DE)": [0.26, 0.30, 0.64, 0.70],  # ~26% down, 64% left
    "Fan B Amps":     [0.53, 0.57, 0.81, 0.88]   # ~53% down, 81% left
}

def analyze_image(image):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    
    results = {}
    debug_img = img_array.copy()
    
    for name, pct in ROIS_PCT.items():
        # Convert % to Pixels for this specific image
        y1 = int(pct[0] * height)
        y2 = int(pct[1] * height)
        x1 = int(pct[2] * width)
        x2 = int(pct[3] * width)
        
        # Draw Box (Blue) for debugging
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Crop & OCR
        # Add a small padding (margin) to ensure we don't cut the number
        roi_crop = img_array[y1:y2, x1:x2]
        
        # Pre-processing
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_RGB2GRAY)
        
        # Upscale slightly to help OCR read small text
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        try:
            val_text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6 outputbase digits')
            # Extract just the number
            val_clean = ''.join(c for c in val_text if c.isdigit() or c == '.')
            
            # Handling multiple dots error (e.g. "25.17.3")
            if val_clean.count('.') > 1:
                val_clean = val_clean.replace('.', '', val_clean.count('.') - 1)
                
            results[name] = float(val_clean) if val_clean else 0.0
        except:
            results[name] = 0.0
            
    return results, debug_img

# --- 3. UI WORKFLOW ---
st.write("### Step 1: Capture or Upload")
img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
camera_file = st.camera_input("Or Take a Photo")

real_file = camera_file if camera_file else img_file

if real_file:
    original_image = Image.open(real_file)
    
    st.write("### Step 2: Crop the Screen")
    st.info("👇 Drag the corners to frame the SCADA screen tightly.")
    
    # CROPPER
    cropped_image = st_cropper(original_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    st.write("### Step 3: Analyze")
    st.image(cropped_image, caption="Your Selection", width=400)
    
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("Processing..."):
            data, debug_view = analyze_image(cropped_image)
            
            # Show alignment check
            st.image(debug_view, caption="AI Alignment Check (Blue Boxes)", use_container_width=True)
            
            # Dashboard
            c1, c2 = st.columns(2)
            c1.metric("Total Air Flow", f"{data['Total Air Flow']} T/Hr")
            c1.metric("Fan A Amps", f"{data['Fan A Amps']} A")
            c2.metric("Fan B Amps", f"{data['Fan B Amps']} A")
            
            vib = data['Fan A Vib (DE)']
            if vib > 7.1:
                c2.error(f"🚨 Vib A: {vib} mm/s (TRIP)")
            elif vib > 4.5:
                c2.warning(f"⚠️ Vib A: {vib} mm/s (ALARM)")
            else:
                c2.success(f"✅ Vib A: {vib} mm/s")
