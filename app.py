import streamlit as st
import pytesseract
from PIL import Image, ImageOps
import numpy as np
import cv2
import shutil
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")
st.title("🏭 Smart Plant Monitor (Zone Search)")

# --- 1. SETUP ---
if not shutil.which("tesseract"):
    st.error("❌ CRITICAL ERROR: Tesseract is missing! Check packages.txt")
    st.stop()

# --- 2. ADVANCED SEARCH ENGINE ---
def find_number_in_zone(df, anchor_keyword, search_direction, search_range_x, search_range_y=30):
    """
    Finds a keyword, then defines a 'Search Zone' relative to it (Left/Right)
    and grabs the largest number found in that zone.
    """
    # Find all occurrences of the keyword
    matches = df[df['text'].str.contains(anchor_keyword, case=False, na=False)]
    
    if matches.empty:
        return 0.0, None

    # We iterate through matches (in case "Amps" appears multiple times)
    # and try to find a valid number near any of them.
    for index, label in matches.iterrows():
        label_x_start = label['left']
        label_x_end = label['left'] + label['width']
        label_y_center = label['top'] + (label['height'] / 2)

        # Define the Search Zone (Rectangle) based on direction
        if search_direction == 'right':
            # Look starting from the end of the label, going right
            zone_x_min = label_x_end
            zone_x_max = label_x_end + search_range_x
        elif search_direction == 'left':
            # Look starting from the left of the label, going left
            zone_x_min = label_x_start - search_range_x
            zone_x_max = label_x_start
        
        # Filter for candidates inside this zone
        # Logic: 
        # 1. Horizontal: Must be within zone_x_min and zone_x_max
        # 2. Vertical: Must be roughly on the same line (within search_range_y)
        candidates = df[
            (df['left'] + (df['width']/2) > zone_x_min) & 
            (df['left'] + (df['width']/2) < zone_x_max) &
            (abs((df['top'] + df['height']/2) - label_y_center) < search_range_y)
        ]

        # Scan candidates for a number
        for i, row in candidates.iterrows():
            text = str(row['text']).strip()
            # Clean: keep digits and dots
            clean_val = ''.join(c for c in text if c.isdigit() or c == '.')
            
            # Validation: Must be a valid float
            if clean_val and len(clean_val) > 1: # Ignore single digits like "A" or "F"
                try:
                    # Fix "double dot" error (e.g., 25.17.3)
                    if clean_val.count('.') > 1:
                        clean_val = clean_val.replace('.', '', clean_val.count('.') - 1)
                    
                    val = float(clean_val)
                    # Success! Return the value and the box of the FOUND NUMBER (not the label)
                    return val, (row['left'], row['top'], row['width'], row['height'])
                except:
                    continue
                    
    return 0.0, None

def process_image(image):
    # Convert to numpy
    img_array = np.array(image)
    
    # 1. Advanced Pre-processing (CLAHE)
    # This balances lighting to make the bottom of the screen as bright as the top
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Resize & Threshold
    resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY_INV)

    # 2. Run OCR
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type='data.frame')
    
    # Clean Data
    data = data[data.conf != -1]
    data['text'] = data['text'].astype(str).str.strip()
    data = data[data['text'] != '']

    # 3. Intelligent Split
    # We split the screen into Top (Common/Fan A) and Bottom (Fan B)
    midpoint = data['top'].max() / 2
    top_data = data[data['top'] < midpoint]
    bot_data = data[data['top'] >= midpoint]

    results = {}
    debug_boxes = []

    # --- SEARCH LOGIC ---
    
    # A. Total Air Flow (Top Data)
    # Look for "FLOW", search 300px to the RIGHT
    val, box = find_number_in_zone(top_data, "FLOW", 'right', 300)
    results["Total Air Flow"] = val
    if box: debug_boxes.append(box)

    # B. Fan A Amps (Top Data)
    # Look for "Amps", search 200px to the LEFT
    val, box = find_number_in_zone(top_data, "Amps", 'left', 200)
    results["Fan A Amps"] = val
    if box: debug_boxes.append(box)

    # C. Fan B Amps (Bottom Data)
    # Look for "Amps", search 200px to the LEFT
    val, box = find_number_in_zone(bot_data, "Amps", 'left', 200)
    results["Fan B Amps"] = val
    if box: debug_boxes.append(box)

    # D. Fan A Vib (Top Data)
    # Look for "mm/se", search 200px to the LEFT
    val, box = find_number_in_zone(top_data, "mm/se", 'left', 200)
    results["Fan A Vib"] = val
    if box: debug_boxes.append(box)

    return results, debug_boxes, resized

# --- 3. UI ---
st.write("### 📸 Upload Plant Photo")
img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Original", width=400)
    
    if st.button("🚀 Run Zone Analysis"):
        with st.spinner("Applying CLAHE & Searching Zones..."):
            results, boxes, processed_img = process_image(image)
            
            # Draw Debug Boxes
            debug_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            for (x, y, w, h) in boxes:
                # Draw Green Box around found numbers
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            
            st.image(debug_img, caption="Green Boxes = Numbers Detected", use_container_width=True)
            
            # Dashboard
            c1, c2 = st.columns(2)
            c1.metric("Total Air Flow", f"{results.get('Total Air Flow', 0)} T/Hr")
            c1.metric("Fan A Amps", f"{results.get('Fan A Amps', 0)} A")
            c2.metric("Fan B Amps", f"{results.get('Fan B Amps', 0)} A")
            c2.metric("Fan A Vib", f"{results.get('Fan A Vib', 0)} mm/s")
            
            # Diagnostics
            st.write("---")
            st.caption("Debugging: If values are 0.0, the AI sees the label but not the number next to it. Check lighting.")
