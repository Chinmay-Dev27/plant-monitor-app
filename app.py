import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")

st.title("🏭 Plant Efficiency & Safety Monitor")
st.write("Upload a photo of the FD Fan SCADA Screen")

# --- 1. SETUP OCR ---
@st.cache_resource
def load_model():
    return easyocr.Reader(['en'])

reader = load_model()

# --- 2. COORDINATE MAPPING (ROI) ---
# Format: [y_min, y_max, x_min, x_max] based on a 1000x666 reference image
ROIS = {
    "Total Air Flow": [220, 255, 175, 260],
    "Fan A Amps": [275, 300, 840, 910],
    "Fan A Vib (DE)": [345, 365, 740, 810],
    "Fan B Amps": [520, 545, 830, 900]
}

def extract_value_from_roi(image, roi_coords):
    # Crop the image to the specific box
    y1, y2, x1, x2 = roi_coords
    
    # Safety check to ensure we don't crop outside image
    if y2 > image.shape[0] or x2 > image.shape[1]:
        return "Error: Image too small"
        
    cropped = image[y1:y2, x1:x2]
    
    # Read text from the cropped area
    result = reader.readtext(cropped, detail=0)
    
    # Return the first number found, or 0.0 if empty
    if result:
        # cleanup: remove non-numeric chars except dot
        clean_text = ''.join(filter(lambda x: x.isdigit() or x == '.', result[0]))
        try:
            return float(clean_text)
        except:
            return 0.0
    return 0.0

# --- 3. MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to standard image format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Resize image to match our coordinate system (1000 width)
    # This ensures the ROIs match even if you take a photo with a different camera
    aspect_ratio = img_array.shape[0] / img_array.shape[1]
    target_width = 1000
    target_height = int(target_width * aspect_ratio)
    img_resized = cv2.resize(img_array, (target_width, target_height))

    st.image(image, caption='Uploaded SCADA Screen', use_column_width=True)
    
    st.subheader("📊 Live Analysis")
    
    # Process Data
    with st.spinner('Scanning SCADA parameters...'):
        
        # Extract values
        flow = extract_value_from_roi(img_resized, ROIS["Total Air Flow"])
        amps_a = extract_value_from_roi(img_resized, ROIS["Fan A Amps"])
        vib_a = extract_value_from_roi(img_resized, ROIS["Fan A Vib (DE)"])
        amps_b = extract_value_from_roi(img_resized, ROIS["Fan B Amps"])
        
        # Display Results in Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Total Air Flow:** {flow} T/Hr")
            st.metric(label="Fan A Amps", value=amps_a)
            st.metric(label="Fan B Amps", value=amps_b)
            
        with col2:
            # VIBRATION LOGIC
            vib_color = "normal"
            if vib_a > 7.1:
                vib_color = "inverse" # Red highlighting equivalent
                st.error(f"⚠️ FAN A VIBRATION: {vib_a} mm/s (TRIP ZONE)")
            elif vib_a > 4.5:
                st.warning(f"⚠️ FAN A VIBRATION: {vib_a} mm/s (ALARM)")
            else:
                st.success(f"✅ Fan A Vibration: {vib_a} mm/s")

    # --- 4. EXPERT TIPS SECTION ---
    st.markdown("---")
    st.subheader("💡 Expert Recommendations")
    
    tips = []
    
    # Logic: Imbalance
    if abs(amps_a - amps_b) > 5.0:
        tips.append(f"🔴 **Load Imbalance Detected:** Fan A and B differ by {round(abs(amps_a - amps_b), 2)} Amps. Check blade pitch auto-sync.")
        
    # Logic: Vibration
    if vib_a > 4.5:
        tips.append("🔴 **High Vibration on Fan A:** Check foundation bolts, coupling alignment, and impeller cleanliness.")
    
    if not tips:
        st.write("✅ System is operating within normal parameters.")
    else:
        for tip in tips:
            st.write(tip)

