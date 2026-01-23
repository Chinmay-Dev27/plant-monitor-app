import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import cv2
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Auto-Align Monitor", layout="wide", page_icon="🏭")
st.title("🏭 Plant Monitor (Auto-Align Tech)")

# --- 1. YOUR MASTER COORDINATES (From your list) ---
# You only need to map these ONCE on your "Reference Image"
ROIS_PCT = {
    "Fuel Flow": [0.07, 0.11, 0.10, 0.17],
    "Air Flow": [0.11, 0.15, 0.10, 0.17],
    "Load": [0.02, 0.06, 0.62, 0.69],
    "FD Fan-A Amps": [0.10, 0.14, 0.48, 0.54],
    "FD Fan-A Vib": [0.62, 0.66, 0.27, 0.33], 
    "FD Fan-B Amps": [0.38, 0.42, 0.48, 0.54],
    "FD Fan-B Vib": [0.62, 0.66, 0.62, 0.68],
    # ... Paste the rest of your list here ...
}

if not shutil.which("tesseract"):
    st.error("❌ Tesseract is missing! Check packages.txt")
    st.stop()

# --- 2. THE ALIGNMENT ENGINE ---
def align_images(image, reference):
    """
    Warps 'image' to match the perspective of 'reference'.
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    
    # MAX_FEATURES: Higher = more accuracy, slower speed
    orb = cv2.ORB_create(MAX_FEATURES=1000)
    
    # Find keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(ref_gray, None)
    
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score (best matches first)
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Remove bad matches (keep top 20%)
    numGoodMatches = int(len(matches) * 0.20)
    matches = matches[:numGoodMatches]
    
    if len(matches) < 4:
        return None, "Not enough features found to align."

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find Homography (The Magic Warp Matrix)
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    if h is None:
        return None, "Could not compute perspective warp."
        
    # Use homography to warp image
    height, width, _ = reference.shape
    aligned_img = cv2.warpPerspective(image, h, (width, height))
    
    return aligned_img, None

def analyze_data(image):
    h, w, _ = image.shape
    results = {}
    
    for name, coords in ROIS_PCT.items():
        # Convert % to pixels
        y1, y2, x1, x2 = int(coords[0]*h), int(coords[1]*h), int(coords[2]*w), int(coords[3]*w)
        
        # Crop & OCR
        if y2 <= h and x2 <= w:
            crop = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            try:
                val = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6 outputbase digits')
                clean = ''.join(c for c in val if c.isdigit() or c == '.')
                if clean.count('.') > 1: clean = clean.replace('.', '', clean.count('.') - 1)
                results[name] = float(clean) if clean else 0.0
            except:
                results[name] = 0.0
        else:
            results[name] = 0.0
            
    return results

# --- 3. UI WORKFLOW ---
st.write("### 🛠️ Setup: Reference Image")
st.info("Upload the PERFECT screenshot you used to map the coordinates. The app will align all future photos to this one.")

ref_file = st.file_uploader("Upload REFERENCE Image (Master)", type=['jpg', 'png'])

if ref_file:
    ref_image = np.array(Image.open(ref_file))
    st.image(ref_image, caption="Master Reference Loaded", width=300)
    
    st.divider()
    
    st.write("### 📸 Daily Operation")
    live_file = st.camera_input("Take a photo of the screen (Angle doesn't matter!)")
    
    if live_file:
        live_image = np.array(Image.open(live_file))
        
        with st.spinner("Auto-Aligning Image..."):
            # 1. Align
            aligned_img, err = align_images(live_image, ref_image)
            
            if aligned_img is not None:
                # Show the user the magic
                st.image(aligned_img, caption="✅ Auto-Corrected View (Aligned)", use_container_width=True)
                
                # 2. Analyze
                data = analyze_data(aligned_img)
                
                # 3. Dashboard
                st.subheader("📊 Live Data")
                c1, c2, c3 = st.columns(3)
                c1.metric("Load", f"{data.get('Load', 0)} MW")
                c2.metric("Air Flow", f"{data.get('Air Flow', 0)} T/Hr")
                c3.metric("Fuel Flow", f"{data.get('Fuel Flow', 0)} T/Hr")
                
                st.subheader("⚠️ Fan Health")
                k1, k2 = st.columns(2)
                
                vib_a = data.get('FD Fan-A Vib', 0)
                k1.metric("Fan A Vib", f"{vib_a} mm/s")
                if vib_a > 4.5: k1.error("High Vibration!")
                
                vib_b = data.get('FD Fan-B Vib', 0)
                k2.metric("Fan B Vib", f"{vib_b} mm/s")
                
                with st.expander("See Raw Data"):
                    st.json(data)
                    
            else:
                st.error(f"Alignment Failed: {err}. Try moving closer to the screen.")
