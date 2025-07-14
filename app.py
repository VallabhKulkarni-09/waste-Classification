"""
Main application module for waste classification system.
"""
import sys
import subprocess
from pathlib import Path

def install_requirements() -> None:
    """Install required packages if not available."""
    try:
        import yolov5
    except ImportError:
        print("Installing YOLOv5... This may take a moment.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yolov5"])
        print("YOLOv5 installed successfully!")

# Install requirements
install_requirements()

import streamlit as st
from src.model import WasteClassifier
from src.ui import (
    setup_page_config, setup_custom_styles, setup_sidebar,
    display_model_info, display_setup_instructions,
    handle_image_upload, handle_camera_input, handle_video_upload
)

# Model path - Change this to your actual model path
MODEL_PATH = "best.pt"

def main() -> None:
    """Main application entry point."""
    # Initialize UI
    setup_page_config()
    setup_custom_styles()
    
    # App title
    st.markdown("<h1 class='main-header'>🗑️ Waste Classification System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: #666;'>Classify waste as Dry or Wet using YOLOv5</p>", unsafe_allow_html=True)
    
    # Initialize model
    classifier = WasteClassifier(MODEL_PATH)
    model = classifier.load_model()
    
    # Setup sidebar and get confidence threshold
    confidence_threshold = setup_sidebar(MODEL_PATH)
    display_model_info(model)
    
    if model is None:
        display_setup_instructions(MODEL_PATH)
        return
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose Input Method",
        ["Upload Image", "Camera Input", "Video Upload"],
        help="Select how you want to provide input"
    )
    
    # Create detection function with current confidence threshold
    def detect_fn(image):
        return classifier.detect(image, confidence_threshold)
    
    # Process based on input method
    if input_method == "Upload Image":
        handle_image_upload(detect_fn)
    elif input_method == "Camera Input":
        handle_camera_input(detect_fn)
    elif input_method == "Video Upload":
        handle_video_upload(detect_fn)

if __name__ == "__main__":
    main()
