import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import pytesseract
import numpy as np
import cv2
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")

# --- 1. SETUP ---
st.title("🏭 Plant Efficiency & Safety Monitor")

if not shutil.which("tesseract"):
    st.error("❌ CRITICAL ERROR: Tesseract is missing! Check packages.txt")
    st.stop()

# --- 2. SETTINGS ---
# The logic expects a standard 1000px wide SCADA screen
TARGET_WIDTH = 1000 

ROIS = {
    "Total Air Flow": [220, 255, 175, 260],
    "Fan A Amps": [275, 300, 840, 910],
    "Fan A Vib (DE)": [345, 365, 740, 810],
    "Fan B Amps": [520, 545, 830, 900]
}

def analyze_image(image):
    # Resize to standard width so coordinates match
    aspect_ratio = image.height / image.width
    new_height = int(TARGET_WIDTH * aspect_ratio)
    img_resized = image.resize((TARGET_WIDTH, new_height))
    img_array = np.array(img_resized)
    
    results = {}
    
    # Draw debug boxes
    debug_img = img_array.copy()
    
    for name, coords in ROIS.items():
        y1, y2, x1, x2 = coords
        
        # Safety check
        if y2 > img_array.shape[0] or x2 > img_array.shape[1]:
            results[name] = 0.0
            continue

        # Draw Box (Blue)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Crop & OCR
        roi_crop = img_array[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        try:
            val_text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6 outputbase digits')
            val_clean = ''.join(c for c in val_text if c.isdigit() or c == '.')
            results[name] = float(val_clean) if val_clean else 0.0
        except:
            results[name] = 0.0
            
    return results, debug_img

# --- 3. UI WORKFLOW ---
st.write("### Step 1: Capture or Upload")
img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
camera_file = st.camera_input("Or Take a Photo")

# Prioritize camera if used, otherwise upload
real_file = camera_file if camera_file else img_file

if real_file:
    original_image = Image.open(real_file)
    
    st.write("### Step 2: Crop the Screen")
    st.info("👇 Drag the corners of the box to select ONLY the SCADA screen (exclude the monitor frame).")
    
    # THE CROPPER WIDGET
    # realtime_update=True shows the crop as you drag
    cropped_image = st_cropper(original_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    st.write("### Step 3: Analyze")
    st.image(cropped_image, caption="Your Selection", width=400)
    
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("Processing..."):
            data, debug_view = analyze_image(cropped_image)
            
            # Show alignment check
            st.image(debug_view, caption="AI Alignment Check (Red Boxes)", use_container_width=True)
            
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
