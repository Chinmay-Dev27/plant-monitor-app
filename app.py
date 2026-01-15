import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")

st.title("🏭 Plant Efficiency & Safety Monitor")

# --- 1. SETUP & DEBUGGING ---
# check if tesseract is actually installed in the system
import shutil
tesseract_cmd = shutil.which("tesseract")
if tesseract_cmd is None:
    st.warning("⚠️ Tesseract binary not found! Did you create packages.txt?")

# Coordinate Mapping (Based on your SCADA screen)
ROIS = {
    "Total Air Flow": [220, 255, 175, 260],
    "Fan A Amps": [275, 300, 840, 910],
    "Fan A Vib (DE)": [345, 365, 740, 810],
    "Fan B Amps": [520, 545, 830, 900]
}

def extract_value_from_roi(image, roi_coords, label):
    y1, y2, x1, x2 = roi_coords
    
    # Safety Check: Image size
    if y2 > image.shape[0] or x2 > image.shape[1]:
        return 0.0, f"Error: Image too small for {label}"

    cropped = image[y1:y2, x1:x2]
    
    # Image Pre-processing
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    try:
        # Tesseract Configuration
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Clean text
        clean_text = ''.join(c for c in text if c.isdigit() or c == '.')
        return float(clean_text) if clean_text else 0.0, None
    except Exception as e:
        return 0.0, str(e)

# --- 2. MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload SCADA Screen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.info("✅ Image received! Processing...") # Debug message
    
    try:
        # Load Image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Resize safely to fixed width (matches your coordinates)
        target_width = 1000
        aspect_ratio = img_array.shape[0] / img_array.shape[1]
        target_height = int(target_width * aspect_ratio)
        img_resized = cv2.resize(img_array, (target_width, target_height))
        
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Run Extraction
        with st.spinner('Reading Data...'):
            flow, err1 = extract_value_from_roi(img_resized, ROIS["Total Air Flow"], "Flow")
            amps_a, err2 = extract_value_from_roi(img_resized, ROIS["Fan A Amps"], "Amps A")
            vib_a, err3 = extract_value_from_roi(img_resized, ROIS["Fan A Vib (DE)"], "Vib A")
            amps_b, err4 = extract_value_from_roi(img_resized, ROIS["Fan B Amps"], "Amps B")

            # Check for errors
            if any([err1, err2, err3, err4]):
                st.error(f"OCR Error: {err1 or err2 or err3 or err4}")
            else:
                # Display Dashboard
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Air Flow", f"{flow} T/Hr")
                    st.metric("Fan A Amps", f"{amps_a} A")
                    st.metric("Fan B Amps", f"{amps_b} A")
                with col2:
                    if vib_a > 7.1:
                        st.error(f"🚨 Vib A: {vib_a} mm/s (TRIP)")
                    elif vib_a > 4.5:
                        st.warning(f"⚠️ Vib A: {vib_a} mm/s (ALARM)")
                    else:
                        st.success(f"✅ Vib A: {vib_a} mm/s")
                
                # Logic Recommendations
                st.markdown("---")
                if abs(amps_a - amps_b) > 5:
                    st.warning(f"💡 Recommendation: Check load imbalance ({abs(amps_a-amps_b):.1f} A diff)")

    except Exception as main_error:
        st.error(f"System Crash: {main_error}")
