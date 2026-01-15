import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
import shutil
import pandas as pd

st.set_page_config(page_title="Calibration Tool", page_icon="🔧")
st.title("🔧 Sequence Calibration Tool")
st.info("Upload an image. I will number every value I see from Top to Bottom.")

if not shutil.which("tesseract"):
    st.error("Tesseract not found.")
    st.stop()

def get_sequence(image):
    img_array = np.array(image)
    
    # 1. Pre-processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY_INV)

    # 2. Get Data
    custom_config = r'--oem 3 --psm 6'
    df = pytesseract.image_to_data(thresh, config=custom_config, output_type='data.frame')
    
    # 3. Filter for Numbers Only
    df = df[df.conf != -1]
    df['text'] = df['text'].astype(str).str.strip()
    
    valid_rows = []
    for i, row in df.iterrows():
        text = row['text']
        # Clean: keep digits and dots
        clean = ''.join(c for c in text if c.isdigit() or c == '.')
        # Check if it looks like a SCADA value (must have digits, maybe a dot)
        if len(clean) > 0 and any(c.isdigit() for c in clean):
             # Save the cleaned text and coordinates
            valid_rows.append({
                'text': clean,
                'x': row['left'],
                'y': row['top'],
                'w': row['width'],
                'h': row['height']
            })
    
    # 4. SORTING (Critical Step)
    # We sort by Y (Top to Bottom), then X (Left to Right)
    # To handle slight misalignment, we round Y to the nearest 20 pixels
    results = pd.DataFrame(valid_rows)
    if not results.empty:
        results['y_group'] = (results['y'] // 50) * 50
        results = results.sort_values(by=['y_group', 'x'])
    
    return results, resized

img_file = st.file_uploader("Upload SCADA Screen", type=['jpg', 'png', 'jpeg'])

if img_file:
    image = Image.open(img_file)
    df_sequence, processed_img = get_sequence(image)
    
    if not df_sequence.empty:
        # Draw the Sequence Numbers on the image
        # We need to convert back to color to draw red boxes
        debug_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        
        st.write("### Detect Sequence:")
        
        # Draw loop
        for idx, row in df_sequence.reset_index().iterrows():
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            val = row['text']
            
            # Draw Box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw Sequence Number (Large Red Text)
            # We put the Index ID (0, 1, 2...) right next to the box
            cv2.putText(debug_img, str(idx), (x - 40, y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        st.image(debug_img, caption="Red Number = Sequence Index", use_container_width=True)
        
        # Show the list as a table so you can copy it
        st.write("### Mapped Values:")
        st.dataframe(df_sequence[['text']].reset_index(drop=True).T)
        
    else:
        st.warning("No numbers detected. Try cropping the black borders.")
