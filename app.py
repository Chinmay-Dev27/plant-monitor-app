import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import cv2
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")

# --- 1. SETUP & DIAGNOSTICS ---
st.title("🏭 Plant Efficiency & Safety Monitor")

# Check if Tesseract is installed on the server
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    st.error("❌ CRITICAL ERROR: Tesseract is missing!")
    st.info("💡 FIX: Create a file named 'packages.txt' in your GitHub and add 'tesseract-ocr' inside it.")
    st.stop() # Stop the app if Tesseract is missing

# --- 2. OPTIMIZED SETTINGS ---
# We shrink images to this width to save RAM (Free Tier Limit)
MAX_WIDTH = 800 

# Coordinates (Scaled to 1000px width reference)
ROIS = {
    "Total Air Flow": [220, 255, 175, 260],
    "Fan A Amps": [275, 300, 840, 910],
    "Fan A Vib (DE)": [345, 365, 740, 810],
    "Fan B Amps": [520, 545, 830, 900]
}

def extract_value_from_roi(image, roi_coords, label):
    y1, y2, x1, x2 = roi_coords
    
    # Check bounds
    if y2 > image.shape[0] or x2 > image.shape[1]:
        return 0.0
        
    cropped = image[y1:y2, x1:x2]
    
    # Pre-processing (Gray -> Threshold)
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    try:
        # Run Tesseract (Digits only mode)
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        clean_text = ''.join(c for c in text if c.isdigit() or c == '.')
        return float(clean_text) if clean_text else 0.0
    except:
        return 0.0

# --- 3. MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload SCADA Screen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.write("🔄 Step 1: Image Received. Resizing...") # Debug Step
    
    try:
        # Load and resize IMMEDIATELY to save memory
        image = Image.open(uploaded_file)
        
        # Resize logic
        aspect_ratio = image.height / image.width
        new_height = int(MAX_WIDTH * aspect_ratio)
        image = image.resize((MAX_WIDTH, new_height))
        
        # Convert to numpy for OpenCV
        img_array = np.array(image)
        
        # Show image (using the new correct parameter)
        st.image(image, caption='Processing...', use_container_width=True)
        
        st.write("🔄 Step 2: Running OCR (This takes ~5 seconds)...") # Debug Step

        # IMPORTANT: We must scale our ROIs because we resized the image to 800px
        # The ROIs were designed for 1000px. So we multiply by 0.8
        scale_factor = MAX_WIDTH / 1000.0
        
        with st.spinner('Analyzing...'):
            results = {}
            for name, coords in ROIS.items():
                # Scale coordinates dynamically
                scaled_coords = [int(c * scale_factor) for c in coords]
                val = extract_value_from_roi(img_array, scaled_coords, name)
                results[name] = val
            
            st.success("✅ Analysis Complete!")
            
            # --- DASHBOARD UI ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Air Flow", f"{results['Total Air Flow']} T/Hr")
                st.metric("Fan A Amps", f"{results['Fan A Amps']} A")
                st.metric("Fan B Amps", f"{results['Fan B Amps']} A")
            with col2:
                vib = results['Fan A Vib (DE)']
                if vib > 7.1:
                    st.error(f"🚨 Vib A: {vib} mm/s (TRIP)")
                elif vib > 4.5:
                    st.warning(f"⚠️ Vib A: {vib} mm/s (ALARM)")
                else:
                    st.success(f"✅ Vib A: {vib} mm/s")

            # --- RECOMMENDATIONS ---
            st.markdown("---")
            tips = []
            amps_a = results['Fan A Amps']
            amps_b = results['Fan B Amps']
            vib_a = results['Fan A Vib (DE)']

            if abs(amps_a - amps_b) > 5.0:
                 st.warning(f"💡 **Imbalance:** {abs(amps_a - amps_b):.1f} A difference between fans.")
            
            if vib_a > 4.5:
                st.error("💡 **Vibration High:** Check coupling/bolts.")
                
            if not tips and abs(amps_a - amps_b) <= 5 and vib_a <= 4.5:
                st.info("System looks normal.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
