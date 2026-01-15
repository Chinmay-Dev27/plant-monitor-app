import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")

st.title("🏭 Plant Efficiency & Safety Monitor")
st.write("Upload a photo of the FD Fan SCADA Screen")

# --- 1. COORDINATE MAPPING (ROI) ---
# Format: [y_min, y_max, x_min, x_max] based on 1000x666 reference
ROIS = {
    "Total Air Flow": [220, 255, 175, 260],
    "Fan A Amps": [275, 300, 840, 910],
    "Fan A Vib (DE)": [345, 365, 740, 810],
    "Fan B Amps": [520, 545, 830, 900]
}

def extract_value_from_roi(image, roi_coords):
    # Crop the image
    y1, y2, x1, x2 = roi_coords
    if y2 > image.shape[0] or x2 > image.shape[1]:
        return 0.0
        
    cropped = image[y1:y2, x1:x2]
    
    # Pre-processing to make text clearer for Tesseract
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    # Thresholding turns grey pixels into pure black or white
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Run Tesseract OCR
    # config: search for numbers only
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # Cleanup text to get float
    try:
        # Remove anything that isn't a digit or dot
        clean_text = ''.join(c for c in text if c.isdigit() or c == '.')
        return float(clean_text)
    except:
        return 0.0

# --- 2. MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Resize to standard width (1000px) so ROIs match
    aspect_ratio = img_array.shape[0] / img_array.shape[1]
    target_width = 1000
    target_height = int(target_width * aspect_ratio)
    img_resized = cv2.resize(img_array, (target_width, target_height))

    st.image(image, caption='Uploaded SCADA Screen', use_column_width=True)
    
    st.subheader("📊 Live Analysis")
    
    with st.spinner('Scanning SCADA parameters...'):
        # Extract values
        flow = extract_value_from_roi(img_resized, ROIS["Total Air Flow"])
        amps_a = extract_value_from_roi(img_resized, ROIS["Fan A Amps"])
        vib_a = extract_value_from_roi(img_resized, ROIS["Fan A Vib (DE)"])
        amps_b = extract_value_from_roi(img_resized, ROIS["Fan B Amps"])
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Air Flow:** {flow} T/Hr")
            st.metric(label="Fan A Amps", value=amps_a)
            st.metric(label="Fan B Amps", value=amps_b)
            
        with col2:
            if vib_a > 7.1:
                st.error(f"⚠️ FAN A VIBRATION: {vib_a} mm/s (TRIP)")
            elif vib_a > 4.5:
                st.warning(f"⚠️ FAN A VIBRATION: {vib_a} mm/s (ALARM)")
            else:
                st.success(f"✅ Fan A Vibration: {vib_a} mm/s")

    # --- 3. EXPERT TIPS ---
    st.markdown("---")
    st.subheader("💡 Expert Recommendations")
    
    tips = []
    if abs(amps_a - amps_b) > 5.0:
        tips.append(f"🔴 **Load Imbalance:** {round(abs(amps_a - amps_b), 2)} Amps diff. Check blade pitch.")
    if vib_a > 4.5:
        tips.append("🔴 **High Vibration:** Check foundation bolts & impeller.")
    
    if not tips:
        st.write("✅ System Normal.")
    else:
        for tip in tips:
            st.write(tip)
