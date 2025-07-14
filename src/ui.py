"""
User interface components for waste classification application.
"""
from typing import Callable, Any, Optional
import streamlit as st
import tempfile
import os
from PIL import Image
import cv2
import numpy as np

def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Waste Classification - Dry vs Wet",
        page_icon="🗑️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_custom_styles() -> None:
    """Apply custom CSS styles."""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            color: #1f77b4;
        }
        .detection-box {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .dry-waste {
            background-color: #e8f5e8;
            border: 2px solid #4caf50;
        }
        .wet-waste {
            background-color: #fff3e0;
            border: 2px solid #ff9800;
        }
        .confidence-score {
            font-size: 1.2em;
            font-weight: bold;
            text-align: center;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar(model_path: str) -> float:
    """
    Setup sidebar with model info and controls.

    Args:
        model_path: Path to the model file

    Returns:
        Selected confidence threshold
    """
    st.sidebar.header("⚙️ Settings")
    st.sidebar.info(f"Model Path: {model_path}")
    
    return st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )

def display_model_info(model: Any) -> None:
    """
    Display model information in sidebar.

    Args:
        model: Loaded YOLOv5 model
    """
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.write("**Model Classes:**")
        try:
            if hasattr(model, 'names'):
                for idx, class_name in model.names.items():
                    st.sidebar.write(f"- {class_name}")
            else:
                st.sidebar.write("- dry")
                st.sidebar.write("- wet")
        except:
            st.sidebar.write("- dry")
            st.sidebar.write("- wet")
    else:
        st.sidebar.error("❌ Failed to load model")

def display_setup_instructions(model_path: str) -> None:
    """
    Display setup instructions when model fails to load.

    Args:
        model_path: Path to the model file
    """
    st.error("❌ Model could not be loaded!")
    st.markdown(f"""
    ### Setup Instructions:
    1. Make sure your `best.pt` model file is in the correct location: `{model_path}`
    2. Update the `MODEL_PATH` variable in the code if needed
    3. Ensure all dependencies are installed:
       ```bash
       pip install torch torchvision yolov5 streamlit opencv-python pillow numpy pandas
       ```
    
    ### Alternative Installation (if above fails):
    ```bash
    pip install torch torchvision
    pip install git+https://github.com/ultralytics/yolov5.git
    pip install streamlit opencv-python pillow numpy pandas
    ```
    
    ### Model Info:
    - **Expected Classes**: Dry waste, Wet waste
    - **Model Type**: YOLOv5 custom trained
    - **File Format**: .pt (PyTorch model)
    - **Training Framework**: ultralytics/yolov5
    """)

def display_results(detections: Any) -> None:
    """
    Display detection results with styling.

    Args:
        detections: DataFrame containing detection results
    """
    if detections is None or detections.empty:
        st.warning("No waste detected in the image")
        return
    
    from .utils import process_detections
    class_name, confidence, num_detections = process_detections(detections)
    
    if class_name:
        if class_name.lower() == 'dry':
            box_class = "dry-waste"
            emoji = "🟢"
            color = "#4caf50"
        else:
            box_class = "wet-waste"
            emoji = "🟡"
            color = "#ff9800"
        
        st.markdown(f"""
        <div class="detection-box {box_class}">
            <div class="confidence-score">
                {emoji} <span style="color: {color};">{class_name.upper()} WASTE</span>
            </div>
            <div class="confidence-score" style="font-size: 1em;">
                Confidence: {confidence:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Total detections:** {num_detections}")
        
        if len(detections) > 1:
            st.subheader("All Detections")
            for idx, detection in detections.iterrows():
                st.write(f"- {detection['name']}: {detection['confidence']:.2%}")
        
        st.subheader("♻️ Disposal Recommendations")
        if class_name.lower() == 'dry':
            st.success("""
            **Dry Waste Disposal:**
            - Can be recycled or composted
            - Separate paper, plastic, metal, and glass
            - Clean containers before disposal
            - Check local recycling guidelines
            """)
        else:
            st.warning("""
            **Wet Waste Disposal:**
            - Compost if organic (food scraps, garden waste)
            - Dispose in organic waste bin
            - Can be used for biogas production
            - Keep separate from dry waste
            """)

def handle_image_upload(detect_fn: Callable) -> None:
    """
    Handle image upload and detection.

    Args:
        detect_fn: Function to perform detection on image
    """
    st.subheader("📸 Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing waste to classify"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Input Image", width=400)
        
        with col2:
            st.subheader("Detection Results")
            
            with st.spinner("Detecting waste..."):
                annotated_image, detections = detect_fn(image)
            
            if annotated_image is not None:
                st.image(annotated_image, caption="Detected Waste", width=400)
                display_results(detections)
            else:
                st.error("Failed to perform detection")

def handle_camera_input(detect_fn: Callable) -> None:
    """
    Handle camera input and detection.

    Args:
        detect_fn: Function to perform detection on image
    """
    st.subheader("📷 Camera Input")
    
    camera_image = st.camera_input("Take a picture of waste")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            st.image(image, caption="Camera Input", width=400)
        
        with col2:
            st.subheader("Detection Results")
            
            with st.spinner("Detecting waste..."):
                annotated_image, detections = detect_fn(image)
            
            if annotated_image is not None:
                st.image(annotated_image, caption="Detected Waste", width=400)
                display_results(detections)
            else:
                st.error("Failed to perform detection")

def handle_video_upload(detect_fn: Callable) -> None:
    """
    Handle video upload and frame-by-frame detection.

    Args:
        detect_fn: Function to perform detection on image
    """
    st.subheader("🎥 Video Upload")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for frame-by-frame detection"
    )
    
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        process_video(video_path, detect_fn)
        
        os.unlink(video_path)

def process_video(video_path: str, detect_fn: Callable) -> None:
    """
    Process video frames for detection.

    Args:
        video_path: Path to video file
        detect_fn: Function to perform detection on image
    """
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"Video loaded: {total_frames} frames, {fps:.2f} FPS")
    
    frame_number = st.slider(
        "Select Frame",
        min_value=0,
        max_value=total_frames-1,
        value=0,
        step=1
    )
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Frame {frame_number}")
            st.image(frame_rgb, caption=f"Frame {frame_number}", width=400)
        
        with col2:
            st.subheader("Detection Results")
            
            with st.spinner("Detecting waste..."):
                annotated_image, detections = detect_fn(frame_rgb)
            
            if annotated_image is not None:
                st.image(annotated_image, caption="Detected Waste", width=400)
                display_results(detections)
            else:
                st.error("Failed to perform detection")
    
    cap.release()
