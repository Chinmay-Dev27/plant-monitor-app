import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import pytesseract
import numpy as np
import cv2
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", layout="wide", page_icon="🏭")
st.title("🏭 FD Fan Efficiency & Health Monitor")

# --- 1. CONFIGURATION: YOUR MAPPED PARAMETERS ---
# These are the exact percentage coordinates you mapped.
# The app will look in these exact spots every time.
ROIS_PCT = {
    "Fuel Flow": [0.07, 0.11, 0.10, 0.17],
    "Air Flow": [0.11, 0.15, 0.10, 0.17],
    "Fuel/Air Ratio": [0.15, 0.19, 0.10, 0.17],
    "Fur DP": [0.19, 0.23, 0.10, 0.17],
    "To SCAPH-A DP": [0.30, 0.34, 0.12, 0.19],
    "To SCAPH-A Temp": [0.34, 0.38, 0.12, 0.19],
    "To SCAPH-B DP": [0.49, 0.53, 0.12, 0.19],
    "To SCAPH-B Temp": [0.53, 0.57, 0.12, 0.19],
    "Freq": [0.02, 0.06, 0.38, 0.45],
    "ATM": [0.02, 0.06, 0.48, 0.55],
    "Load": [0.02, 0.06, 0.62, 0.69],
    
    # --- FD FAN A ---
    "FD Fan-A Amps": [0.10, 0.14, 0.48, 0.54],
    "FD Fan-A Loading %": [0.08, 0.12, 0.27, 0.33],
    "FD Fan-A LO Pressure": [0.22, 0.26, 0.23, 0.29],
    "FD Fan-A LOP Temp": [0.42, 0.46, 0.27, 0.33], 
    "FD Fan-A mmSe": [0.62, 0.66, 0.27, 0.33],  # Vibration
    "FD Fan-A Temp1": [0.08, 0.12, 0.34, 0.40],
    
    # --- FD FAN B ---
    "FD Fan-B Amps": [0.38, 0.42, 0.48, 0.54],
    "FD Fan-B Loading %": [0.08, 0.12, 0.62, 0.68],
    "FD Fan-B LO Pressure": [0.22, 0.26, 0.58, 0.64],
    "FD Fan-B LOP Temp": [0.42, 0.46, 0.62, 0.68],
    "FD Fan-B mmSe": [0.62, 0.66, 0.62, 0.68],  # Vibration
    
    # --- CRITICAL UNIT PARAMETERS ---
    "MS TMP": [0.16, 0.20, 0.85, 0.91],
    "MS PR": [0.20, 0.24, 0.85, 0.91],
    "HRH TMP": [0.24, 0.28, 0.85, 0.91],
    "DR LVL": [0.32, 0.36, 0.85, 0.91],
    "FW FL": [0.36, 0.40, 0.85, 0.91],
    "PA HDR": [0.52, 0.56, 0.85, 0.91],
}

# --- 2. SETUP OCR ENGINE ---
if not shutil.which("tesseract"):
    st.error("❌ CRITICAL ERROR: Tesseract is missing! Please ensure 'packages.txt' is in your GitHub repo.")
    st.stop()

def analyze_image(image):
    # Convert image to numpy array for OpenCV
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    results = {}
    
    # Loop through every parameter in your list
    for name, coords in ROIS_PCT.items():
        # 1. Convert Percentage to Pixels
        y1, y2, x1, x2 = int(coords[0]*h), int(coords[1]*h), int(coords[2]*w), int(coords[3]*w)
        
        # 2. Safety Check (Bounds)
        if y2 > h or x2 > w:
            results[name] = 0.0
            continue
            
        # 3. Crop the specific box
        crop = img_array[y1:y2, x1:x2]
        
        # 4. Image Enhancement for OCR (The "Magic" Step)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Resize: Zoom in 3x to make small numbers big and clear
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # Threshold: Convert to pure black and white
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # 5. Run Tesseract OCR
        try:
            # Config: 'digits' ensures we don't accidentally read letters as numbers
            text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6 outputbase digits')
            
            # 6. Cleaning the Result
            # Keep only digits and dots
            clean_text = ''.join(c for c in text if c.isdigit() or c == '.')
            
            # Handle "double dot" error (e.g., reading "25..17")
            if clean_text.count('.') > 1:
                clean_text = clean_text.replace('.', '', clean_text.count('.') - 1)
            
            # Store the result
            results[name] = float(clean_text) if clean_text else 0.0
        except:
            results[name] = 0.0
            
    return results

# --- 3. THE APP INTERFACE ---
# Upload Section
st.write("### 📸 Capture SCADA Screen")
st.info("Upload a photo or take one. Crop exactly to the screen edges.")

img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
camera_file = st.camera_input("Or Take a Photo")
real_file = camera_file if camera_file else img_file

if real_file:
    original = Image.open(real_file)
    
    # Cropper Tool
    st.write("#### ✂️ Step 1: Crop to Edges")
    cropped_img = st_cropper(original, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
    
    st.write("#### 📊 Step 2: Live Analysis")
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("Extracting 30+ Parameters..."):
            data = analyze_image(cropped_img)
            
            # --- DISPLAY DASHBOARD ---
            
            # Row 1: Key Performance Indicators (KPIs)
            st.subheader("🔥 Key Plant KPIs")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Load", f"{data.get('Load', 0)} MW")
            kpi2.metric("Total Air Flow", f"{data.get('Air Flow', 0)} T/Hr")
            kpi3.metric("Fuel Flow", f"{data.get('Fuel Flow', 0)} T/Hr")
            kpi4.metric("Frequency", f"{data.get('Freq', 0)} Hz")
            
            st.divider()
            
            # Row 2: Fan A vs Fan B Comparison
            st.subheader("⚙️ Fan Performance Comparison")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### 🅰️ FD Fan A")
                st.metric("Amps", f"{data.get('FD Fan-A Amps', 0)} A")
                st.metric("Vibration", f"{data.get('FD Fan-A mmSe', 0)} mm/s", 
                          delta_color="inverse" if data.get('FD Fan-A mmSe', 0) > 4.5 else "normal")
                st.metric("Lube Oil Pr", f"{data.get('FD Fan-A LO Pressure', 0)}")
                
                # Health Check Logic A
                if data.get('FD Fan-A mmSe', 0) > 7.1:
                    st.error("🚨 CRITICAL: High Vibration Trip Level!")
                elif data.get('FD Fan-A mmSe', 0) > 4.5:
                    st.warning("⚠️ ALARM: High Vibration Warning")
            
            with col_b:
                st.markdown("### 🅱️ FD Fan B")
                st.metric("Amps", f"{data.get('FD Fan-B Amps', 0)} A")
                st.metric("Vibration", f"{data.get('FD Fan-B mmSe', 0)} mm/s",
                          delta_color="inverse" if data.get('FD Fan-B mmSe', 0) > 4.5 else "normal")
                st.metric("Lube Oil Pr", f"{data.get('FD Fan-B LO Pressure', 0)}")

                # Health Check Logic B
                if data.get('FD Fan-B mmSe', 0) > 7.1:
                    st.error("🚨 CRITICAL: High Vibration Trip Level!")
                
            st.divider()
            
            # Row 3: Full Data Table (Expandable)
            with st.expander("📋 View All Extracted Data"):
                st.json(data)

            # Row 4: Expert Recommendations
            st.subheader("💡 AI Expert Recommendations")
            
            tips = []
            
            # Imbalance Check
            amps_a = data.get('FD Fan-A Amps', 0)
            amps_b = data.get('FD Fan-B Amps', 0)
            if abs(amps_a - amps_b) > 5.0:
                tips.append(f"🔴 **Load Imbalance:** Fan A and B differ by {abs(amps_a-amps_b):.1f} Amps. Check blade pitch sync.")
            
            # Efficiency/Flow Check (Simple heuristic)
            air_flow = data.get('Air Flow', 0)
            load = data.get('Load', 0)
            # Example rule: At 200MW, expect ~600 T/hr. If flow is < 500, check blockage.
            if load > 150 and air_flow < 400:
                tips.append("🟠 **Low Air Flow:** Flow is lower than expected for this Load. Check for APH clogging or Filter blockage.")
            
            if not tips:
                st.success("✅ System is operating within normal parameters.")
            else:
                for tip in tips:
                    st.write(tip)

