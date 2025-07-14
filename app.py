import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import time
import sys
import subprocess

# Install required packages if not available
def install_requirements():
    try:
        import yolov5
    except ImportError:
        st.info("Installing YOLOv5... This may take a moment.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yolov5"])
        st.success("YOLOv5 installed successfully!")
        st.rerun()

# Install requirements
install_requirements()

# Set page config
st.set_page_config(
    page_title="Waste Classification - Dry vs Wet",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Model path - Change this to your model path
MODEL_PATH = "/home/vallabh/Documents/waste app/best.pt"  # Change this to your actual model path

# Load model function
@st.cache_resource
def load_model():
    """Load YOLOv5 model from best.pt file"""
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            st.info("Please update the MODEL_PATH variable in the code with the correct path to your best.pt file")
            return None
        
        # Load the model using torch.hub (original YOLOv5 method)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        model.conf = 0.5  # confidence threshold
        model.iou = 0.45  # IoU threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Try alternative method with yolov5 package
        try:
            import yolov5
            model = yolov5.load(MODEL_PATH)
            return model
        except Exception as e2:
            st.error(f"Alternative loading method also failed: {e2}")
            st.info("Make sure you have the correct model file and all dependencies installed")
            return None

def detect_waste(model, image):
    """Perform detection on image"""
    if model is None:
        return None, None
    
    # Run inference
    results = model(image)
    
    # Get detections - handle both torch.hub and yolov5 package formats
    try:
        # Try torch.hub format first
        detections = results.pandas().xyxy[0]
        annotated_image = np.array(results.render()[0])
    except:
        # Try yolov5 package format
        try:
            detections = results.pandas().xyxy[0]
            annotated_image = np.array(results.render()[0])
        except:
            # Manual processing if pandas method fails
            detections = []
            annotated_image = np.array(image)
            
            # Process results manually
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = box
                class_name = model.names[int(cls)]
                
                detections.append({
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'confidence': conf,
                    'class': int(cls),
                    'name': class_name
                })
            
            # Convert to DataFrame
            import pandas as pd
            detections = pd.DataFrame(detections)
            
            # Simple annotation (draw boxes on image)
            annotated_image = draw_boxes(np.array(image), detections)
    
    return annotated_image, detections

def draw_boxes(image, detections):
    """Draw bounding boxes on image"""
    if detections.empty:
        return image
    
    img = image.copy()
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        class_name = detection['name']
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

def process_detections(detections):
    """Process detections and return classification results"""
    if detections.empty:
        return None, None, 0
    
    # Get the detection with highest confidence
    best_detection = detections.loc[detections['confidence'].idxmax()]
    
    class_name = best_detection['name']
    confidence = best_detection['confidence']
    
    return class_name, confidence, len(detections)

def main():
    # App title
    st.markdown("<h1 class='main-header'>🗑️ Waste Classification System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: #666;'>Classify waste as Dry or Wet using YOLOv5</p>", unsafe_allow_html=True)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Model info
    st.sidebar.info(f"Model Path: {MODEL_PATH}")
    
    # Load model
    model = load_model()
    
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        # Display model info
        st.sidebar.write("**Model Classes:**")
        try:
            # Handle different model formats
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
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose Input Method",
        ["Upload Image", "Camera Input", "Video Upload"],
        help="Select how you want to provide input"
    )
    
    # Main content area
    if model is None:
        st.error("❌ Model could not be loaded!")
        st.markdown(f"""
        ### Setup Instructions:
        1. Make sure your `best.pt` model file is in the correct location: `{MODEL_PATH}`
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
        return
    
    # Process based on input method
    if input_method == "Upload Image":
        handle_image_upload(model, confidence_threshold)
    elif input_method == "Camera Input":
        handle_camera_input(model, confidence_threshold)
    elif input_method == "Video Upload":
        handle_video_upload(model, confidence_threshold)

def handle_image_upload(model, confidence_threshold):
    """Handle image upload and detection"""
    st.subheader("📸 Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing waste to classify"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            # Fixed: Use width parameter instead of use_container_width for compatibility
            st.image(image, caption="Input Image", width=400)
        
        with col2:
            st.subheader("Detection Results")
            
            # Perform detection
            with st.spinner("Detecting waste..."):
                annotated_image, detections = detect_waste_with_confidence(model, image, confidence_threshold)
            
            if annotated_image is not None:
                st.image(annotated_image, caption="Detected Waste", width=400)
                
                # Process and display results
                display_results(detections)
            else:
                st.error("Failed to perform detection")

def handle_camera_input(model, confidence_threshold):
    """Handle camera input for real-time detection"""
    st.subheader("📷 Camera Input")
    
    # Camera input
    camera_image = st.camera_input("Take a picture of waste")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            st.image(image, caption="Camera Input", width=400)
        
        with col2:
            st.subheader("Detection Results")
            
            # Perform detection
            with st.spinner("Detecting waste..."):
                annotated_image, detections = detect_waste_with_confidence(model, image, confidence_threshold)
            
            if annotated_image is not None:
                st.image(annotated_image, caption="Detected Waste", width=400)
                
                # Process and display results
                display_results(detections)
            else:
                st.error("Failed to perform detection")

def handle_video_upload(model, confidence_threshold):
    """Handle video upload for detection"""
    st.subheader("🎥 Video Upload")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for frame-by-frame detection"
    )
    
    if uploaded_video is not None:
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Process video
        process_video(model, video_path, confidence_threshold)
        
        # Clean up
        os.unlink(video_path)

def detect_waste_with_confidence(model, image, confidence_threshold):
    """Perform detection on image with custom confidence threshold"""
    if model is None:
        return None, None
    
    # Set confidence threshold
    model.conf = confidence_threshold
    
    # Run inference
    results = model(image)
    
    # Get detections - handle both torch.hub and yolov5 package formats
    try:
        # Try torch.hub format first
        detections = results.pandas().xyxy[0]
        annotated_image = np.array(results.render()[0])
    except:
        # Try yolov5 package format
        try:
            detections = results.pandas().xyxy[0]
            annotated_image = np.array(results.render()[0])
        except:
            # Manual processing if pandas method fails
            detections = []
            annotated_image = np.array(image)
            
            # Process results manually
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf >= confidence_threshold:  # Apply confidence threshold
                    x1, y1, x2, y2 = box
                    class_name = model.names[int(cls)]
                    
                    detections.append({
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'confidence': conf,
                        'class': int(cls),
                        'name': class_name
                    })
            
            # Convert to DataFrame
            import pandas as pd
            detections = pd.DataFrame(detections)
            
            # Simple annotation (draw boxes on image)
            annotated_image = draw_boxes(np.array(image), detections)
    
    return annotated_image, detections

def process_video(model, video_path, confidence_threshold):
    """Process video frame by frame"""
    cap = cv2.VideoCapture(video_path)
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"Video loaded: {total_frames} frames, {fps:.2f} FPS")
    
    # Frame selection
    frame_number = st.slider(
        "Select Frame",
        min_value=0,
        max_value=total_frames-1,
        value=0,
        step=1
    )
    
    # Process selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Frame {frame_number}")
            st.image(frame_rgb, caption=f"Frame {frame_number}", width=400)
        
        with col2:
            st.subheader("Detection Results")
            
            # Perform detection
            with st.spinner("Detecting waste..."):
                annotated_image, detections = detect_waste_with_confidence(model, frame_rgb, confidence_threshold)
            
            if annotated_image is not None:
                st.image(annotated_image, caption="Detected Waste", width=400)
                
                # Process and display results
                display_results(detections)
            else:
                st.error("Failed to perform detection")
    
    cap.release()

def display_results(detections):
    """Display detection results"""
    if detections is None or detections.empty:
        st.warning("No waste detected in the image")
        return
    
    # Process detections
    class_name, confidence, num_detections = process_detections(detections)
    
    if class_name:
        # Determine waste type styling
        if class_name.lower() == 'dry':
            box_class = "dry-waste"
            emoji = "🟢"
            color = "#4caf50"
        else:
            box_class = "wet-waste"
            emoji = "🟡"
            color = "#ff9800"
        
        # Display main result
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
        
        # Additional details
        st.markdown(f"**Total detections:** {num_detections}")
        
        # Show all detections
        if len(detections) > 1:
            st.subheader("All Detections")
            for idx, detection in detections.iterrows():
                st.write(f"- {detection['name']}: {detection['confidence']:.2%}")
        
        # Disposal recommendations
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

if __name__ == "__main__":
    main()