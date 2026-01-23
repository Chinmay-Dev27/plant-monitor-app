import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import cv2
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Auto-Align Monitor", layout="wide", page_icon="🏭")
st.title("🏭 Plant Monitor (Auto-Align Tech)")

# --- 1. YOUR MASTER COORDINATES ---
# (Paste your Full List of mapped coordinates here)
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
    "FD Fan-A Amps": [0.10, 0.14, 0.48, 0.54],
    "FD Fan-A Loading %": [0.08, 0.12, 0.27, 0.33],
    "FD Fan-A LO Pressure": [0.22, 0.26, 0.23, 0.29],
    "FD Fan-A LOP Temp": [0.42, 0.46, 0.27, 0.33], 
    "FD Fan-A mmSe": [0.62, 0.66, 0.27, 0.33],
    "FD Fan-A Temp1": [0.08, 0.12, 0.34, 0.40],
    "FD Fan-B Amps": [0.38, 0.42, 0.48, 0.54],
    "FD Fan-B Loading %": [0.08, 0.12, 0.62, 0.68],
    "FD Fan-B LO Pressure": [0.22, 0.26, 0.58, 0.64],
    "FD Fan-B LOP Temp": [0.42, 0.46, 0.62, 0.68],
    "FD Fan-B mmSe": [0.62, 0.66, 0.62, 0.68],
}

# --- 2. THE ALIGNMENT ENGINE (FIXED) ---
def align_images(image, reference):
    """
    Warps 'image' to match the perspective of 'reference'.
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    
    # FIX: Use 'nfeatures' instead of MAX_FEATURES
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_gray, None)
    
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove bad matches (keep top 15%)
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
    
    if len(matches) < 4:
        return None, "Not enough features found. Try moving closer."

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find Homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    if h is None:
        return None, "Perspective warp failed."
        
    # Warp image
    height, width, _ = reference.shape
    aligned_img = cv2.warpPerspective(image, h, (width, height))
    
    return aligned_img, None

def analyze_data(image):
    h, w, _ = image.shape
    results = {}
    
    for name, coords in ROIS_PCT.items():
        y1, y2, x1, x2 = int(coords[0]*h), int(coords[1]*h), int(coords[2]*w), int(coords[3]*w)
        
        # Bounds Check
        if y2 > h or x2 > w:
            results[name] = 0.0
            continue

        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        try:
            val = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6 outputbase digits')
            clean = ''.join(c for c in val if c.isdigit() or c == '.')
            if clean.count('.') > 1: clean = clean.replace('.', '', clean.count('.') - 1)
            results[name] = float(clean) if clean else 0.0
        except:
            results[name] = 0.0
            
    return results

# --- 3. UI WORKFLOW ---
st.sidebar.header("⚙️ Setup")
ref_file = st.sidebar.file_uploader("Upload REFERENCE Image (Master)", type=['jpg', 'png'])

if ref_file:
    # Load Master
    ref_image = np.array(Image.open(ref_file))
    st.sidebar.image(ref_image, caption="Master Reference", width=200)
    
    st.write("### 📸 Live Plant Monitor")
    live_file = st.camera_input("Take a photo of the screen")
    
    if live_file:
        # Load and Resize Input (Prevents Memory Crash)
        live_pil = Image.open(live_file)
        # Resize if huge (keep aspect ratio)
        if live_pil.width > 1600:
            ratio = 1600 / live_pil.width
            new_h = int(live_pil.height * ratio)
            live_pil = live_pil.resize((1600, new_h))
            
        live_image = np.array(live_pil)
        
        with st.spinner("Aligning & Analyzing..."):
            aligned_img, err = align_images(live_image, ref_image)
            
            if aligned_img is not None:
                # Show Comparison
                st.image(aligned_img, caption="✅ Auto-Corrected View", width=700)
                
                # Analyze
                data = analyze_data(aligned_img)
                
                # --- DASHBOARD ---
                st.divider()
                st.subheader("📊 Plant KPIs")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Load", f"{data.get('Load', 0)} MW")
                k2.metric("Air Flow", f"{data.get('Air Flow', 0)} T/Hr")
                k3.metric("Fuel Flow", f"{data.get('Fuel Flow', 0)} T/Hr")
                k4.metric("Frequency", f"{data.get('Freq', 0)} Hz")
                
                st.subheader("⚙️ FD Fans Status")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.info("Fan A")
                    st.metric("Amps", f"{data.get('FD Fan-A Amps', 0)} A")
                    vib = data.get('FD Fan-A mmSe', 0)
                    st.metric("Vibration", f"{vib} mm/s")
                    if vib > 4.5: st.error("High Vibration!")

                with c2:
                    st.info("Fan B")
                    st.metric("Amps", f"{data.get('FD Fan-B Amps', 0)} A")
                    vib = data.get('FD Fan-B mmSe', 0)
                    st.metric("Vibration", f"{vib} mm/s")
                    if vib > 4.5: st.error("High Vibration!")
                    
            else:
                st.error(f"⚠️ Alignment Failed: {err}")
                st.warning("Tip: Try to frame the screen so it looks somewhat like your Reference image.")

else:
    st.info("👈 Please upload your 'Master Reference' screenshot in the Sidebar to start.")
