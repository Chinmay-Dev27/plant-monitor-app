import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import shutil

st.set_page_config(layout="wide", page_title="Coordinate Finder")

st.title("🎯 SCADA Coordinate Finder")
st.info("Use this tool to find the EXACT position of your parameters.")

# 1. Upload & Crop
img_file = st.file_uploader("Upload SCADA Screen", type=['jpg', 'png', 'jpeg'])

if img_file:
    # Load Image
    original_image = Image.open(img_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### 1. Crop the Screen")
        # Get the cropped image
        cropped_image = st_cropper(original_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)
        st.caption("Make sure to crop exactly to the edges of the SCADA graphic.")

    with col2:
        st.write("### 2. Find Your Coordinates")
        
        if cropped_image:
            # Convert to numpy for dimensions
            img_array = np.array(cropped_image)
            height, width, _ = img_array.shape
            
            # --- INTERACTIVE SLIDERS ---
            # These sliders represent Percentages (0.0 to 1.0)
            st.write("Adjust these sliders until the **GREEN BOX** covers the number perfectly.")
            
            y_range = st.slider("Vertical Position (Top/Bottom)", 0.0, 1.0, (0.20, 0.30), 0.01)
            x_range = st.slider("Horizontal Position (Left/Right)", 0.0, 1.0, (0.15, 0.25), 0.01)
            
            # Draw the box based on sliders
            debug_img = img_array.copy()
            
            # Convert % to Pixels
            y1 = int(y_range[0] * height)
            y2 = int(y_range[1] * height)
            x1 = int(x_range[0] * width)
            x2 = int(x_range[1] * width)
            
            # Draw Green Box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Show Image
            st.image(debug_img, use_container_width=True)
            
            # --- GENERATE CODE ---
            st.success("### ✅ Copy This Code:")
            code_snippet = f'"{st.text_input("Parameter Name", "Total Air Flow")}": [{y_range[0]:.2f}, {y_range[1]:.2f}, {x_range[0]:.2f}, {x_range[1]:.2f}],'
            st.code(code_snippet, language="python")
