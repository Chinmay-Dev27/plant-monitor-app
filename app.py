import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import cv2
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")

# --- 1. SETUP ---
st.title("🏭 Plant Efficiency & Safety Monitor")

# Check Tesseract
if not shutil.which("tesseract"):
    st.error("❌ CRITICAL ERROR: Tesseract is missing! Check packages.txt")
    st.stop()

# --- 2. SETTINGS ---
MAX_WIDTH = 800 

# Coordinates (Optimized for 1000px width digital screenshot)
ROIS = {
    "Total Air Flow": [220, 255, 175, 260],
    "Fan A Amps": [275, 300, 840, 910],
    "Fan A Vib (DE)": [345, 365, 740, 810],
    "Fan B Amps": [520, 545, 830, 900]
}

def extract_value_from_roi(image, roi_coords):
    y1, y2, x1, x2 = roi_coords
    if y2 > image.shape[0] or x2 > image.shape[1]:
        return 0.0
    
    cropped = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    try:
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        clean_text = ''.join(c for c in text if c.isdigit() or c == '.')
        return float(clean_text) if clean_text else 0.0
    except:
        return 0.0

# --- 3. MAIN APP ---
uploaded_file = st.file_uploader("Upload Cropped SCADA Screen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.write("🔄 Processing...")
    
    try:
        # Load & Resize
        image = Image.open(uploaded_file)
        aspect_ratio = image.height / image.width
        new_height = int(MAX_WIDTH * aspect_ratio)
        image = image.resize((MAX_WIDTH, new_height))
        img_array = np.array(image)

        # Scale Factor
        scale_factor = MAX_WIDTH / 1000.0
        
        # --- DEBUGGING: DRAW BOXES ---
        # We create a copy of the image to draw red rectangles on
        debug_img = img_array.copy()
        
        results = {}
        for name, coords in ROIS.items():
            # Scale coordinates
            scaled_coords = [int(c * scale_factor) for c in coords]
            y1, y2, x1, x2 = scaled_coords
            
            # Draw Red Rectangle (Thickness 2)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Extract
            val = extract_value_from_roi(img_array, scaled_coords)
            results[name] = val
        
        # Display the image WITH boxes so you can see alignment
        st.image(debug_img, caption='Red Boxes show where AI is reading', width=MAX_WIDTH)
        
        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Air Flow", f"{results['Total Air Flow']} T/Hr")
            st.metric("Fan A Amps", f"{results['Fan A Amps']} A")
        with col2:
            st.metric("Fan B Amps", f"{results['Fan B Amps']} A")
            vib = results['Fan A Vib (DE)']
            if vib > 7.1:
                st.error(f"🚨 Vib A: {vib} (TRIP)")
            elif vib > 4.5:
                st.warning(f"⚠️ Vib A: {vib} (ALARM)")
            else:
                st.success(f"✅ Vib A: {vib} mm/s")

    except Exception as e:
        st.error(f"Error: {e}")
