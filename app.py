import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
import shutil
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Plant Monitor AI", page_icon="🏭")
st.title("🏭 Smart Plant Monitor (Auto-Locate)")

# --- 1. SETUP ---
if not shutil.which("tesseract"):
    st.error("❌ CRITICAL ERROR: Tesseract is missing! Check packages.txt")
    st.stop()

# --- 2. INTELLIGENT PARSER ENGINE ---
def find_value_near_label(df, label_keywords, search_area='right', x_limit=300, y_limit=50):
    """
    Scans the OCR data for a specific label and finds the nearest number next to it.
    df: The dataframe containing all text found by OCR.
    label_keywords: A list of words to match (e.g., ["AIR", "FLOW"])
    search_area: 'right' (standard) or 'below' (for tables).
    """
    # 1. Find the Label
    # We look for rows in the data that contain our keywords
    matches = df[df['text'].str.contains(label_keywords[0], case=False, na=False)]
    
    if matches.empty:
        return 0.0, None

    # Get coordinates of the label
    label_x = matches.iloc[0]['left'] + matches.iloc[0]['width'] # Right edge of label
    label_y = matches.iloc[0]['top']
    label_h = matches.iloc[0]['height']

    # 2. Search for the Value
    # We filter for text that is:
    # - To the RIGHT of the label (within x_limit pixels)
    # - On the SAME LINE (within y_limit pixels vertical)
    # - Is a Number
    
    candidates = df[
        (df['left'] > label_x) & 
        (df['left'] < label_x + x_limit) &
        (abs(df['top'] - label_y) < y_limit)
    ]

    for index, row in candidates.iterrows():
        text = str(row['text']).strip()
        # Clean the text to see if it's a number
        clean_val = ''.join(c for c in text if c.isdigit() or c == '.')
        
        # Validation: Must have at least one digit and not be empty
        if clean_val and any(c.isdigit() for c in clean_val):
            try:
                # Handle double dots error (e.g. "723..66")
                if clean_val.count('.') > 1:
                    clean_val = clean_val.replace('.', '', clean_val.count('.') - 1)
                return float(clean_val), (row['left'], row['top'], row['width'], row['height'])
            except:
                continue
                
    return 0.0, None

def process_image(image):
    # Convert to numpy
    img_array = np.array(image)
    
    # 1. Pre-processing (Critical for digital fonts)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Resize: Double the size to make small numbers readable
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Threshold: Make it black and white
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 2. Run OCR on the WHOLE image once
    # output_type='data.frame' gives us a table of every word and its x,y location
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type='data.frame')
    
    # Clean up data (remove empty rows)
    data = data[data.conf != -1]
    data['text'] = data['text'].astype(str).str.strip()
    data = data[data['text'] != '']

    # 3. Smart Search Definitions
    # We map "Variable Name" -> "Keywords to find"
    results = {}
    debug_boxes = [] # To draw on image later

    # Define what we are looking for
    targets = {
        "Total Air Flow": ["FLOW"],      # Looks for "FLOW" (from AIR FLOW)
        "Fan A Amps": ["25."],           # HACK: Looking for the unique value region if label is missing
        "Fan B Amps": ["35."],           # HACK: Looking for unique value region
        # Better approach for identical labels (like "Amps" appearing twice):
        # We can split the image into Top/Bottom halves!
    }
    
    # --- STRATEGY FOR DUPLICATE LABELS (Fan A vs Fan B) ---
    # Since "Amps" appears twice, we split the dataframe into Top and Bottom
    midpoint = data['top'].max() / 2
    
    top_data = data[data['top'] < midpoint]
    bot_data = data[data['top'] >= midpoint]

    # Find Air Flow (Top Left usually)
    val, box = find_value_near_label(top_data, ["FLOW"])
    results["Total Air Flow"] = val
    if box: debug_boxes.append(box)

    # Find Fan A Amps (Top Half, look for 'Amps' label)
    val, box = find_value_near_label(top_data, ["Amps"], search_area='left', x_limit=150) 
    # Note: On your screen, "Amps" is to the RIGHT of the number. 
    # So we actually need to look to the LEFT of the label "Amps".
    
    # Let's fix the logic for "Amps" specifically.
    # The screen says: "25.173 Amps". The label is AFTER the number.
    # So we find "Amps", then look at the word immediately BEFORE it.
    
    # --- REVISED LOGIC FOR "NUMBER BEFORE LABEL" ---
    def find_val_before_label(df, label_keyword):
        matches = df[df['text'].str.contains(label_keyword, case=False)]
        if matches.empty: return 0.0, None
        
        # Get the word index
        idx = matches.index[0]
        # Look at the previous word (index - 1)
        if idx - 1 in df.index:
            prev_word = df.loc[idx-1]
            text = str(prev_word['text'])
            clean_val = ''.join(c for c in text if c.isdigit() or c == '.')
            if clean_val:
                return float(clean_val), (prev_word['left'], prev_word['top'], prev_word['width'], prev_word['height'])
        return 0.0, None

    # Run specific searches
    # Fan A Amps (Top Half)
    val_a, box_a = find_val_before_label(top_data, "Amps")
    results["Fan A Amps"] = val_a
    if box_a: debug_boxes.append(box_a)

    # Fan B Amps (Bottom Half)
    val_b, box_b = find_val_before_label(bot_data, "Amps")
    results["Fan B Amps"] = val_b
    if box_b: debug_boxes.append(box_b)

    # Vibration (This is harder because label is far away). 
    # Let's try searching for "mm/se" which is unique!
    # Fan A Vib (Top Half)
    val_vib_a, box_vib_a = find_val_before_label(top_data, "mm/se")
    results["Fan A Vib"] = val_vib_a
    if box_vib_a: debug_boxes.append(box_vib_a)

    return results, debug_boxes, gray

# --- 3. UI ---
img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Original", width=400)
    
    if st.button("🚀 Analyze with Smart Search"):
        with st.spinner("Reading full screen text..."):
            results, boxes, processed_img = process_image(image)
            
            # Draw Debug Boxes on processed image to show what we found
            debug_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            for (x, y, w, h) in boxes:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            
            st.image(debug_img, caption="Green Boxes = What AI Found", use_container_width=True)
            
            # Dashboard
            c1, c2 = st.columns(2)
            c1.metric("Total Air Flow", f"{results.get('Total Air Flow', 0)} T/Hr")
            c1.metric("Fan A Amps", f"{results.get('Fan A Amps', 0)} A")
            c2.metric("Fan B Amps", f"{results.get('Fan B Amps', 0)} A")
            c2.metric("Fan A Vib", f"{results.get('Fan A Vib', 0)} mm/s")
