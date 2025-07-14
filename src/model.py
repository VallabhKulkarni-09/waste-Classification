"""
Model loading and inference functions for waste classification.
"""
from typing import Tuple, Optional, Union
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from pathlib import Path

class WasteClassifier:
    """YOLOv5 model wrapper for waste classification."""
    
    def __init__(self, model_path: str):
        """
        Initialize the waste classifier.

        Args:
            model_path: Path to the YOLOv5 model file (.pt)
        """
        self.model_path = model_path
        self.model = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45

    @st.cache_resource
    def load_model(_self) -> Optional[torch.nn.Module]:
        """
        Load YOLOv5 model from file.

        Returns:
            Loaded PyTorch model or None if loading fails
        """
        try:
            if not os.path.exists(_self.model_path):
                st.error(f"Model file not found: {_self.model_path}")
                return None
            
            # Load model using torch.hub
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                 path=_self.model_path, force_reload=True)
            model.conf = _self.conf_threshold
            model.iou = _self.iou_threshold
            return model
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            # Try alternative method
            try:
                import yolov5
                model = yolov5.load(_self.model_path)
                return model
            except Exception as e2:
                st.error(f"Alternative loading method also failed: {e2}")
                return None

    def detect(self, image: Union[np.ndarray, Image.Image], 
               confidence_threshold: float = None) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """
        Perform waste detection on an image.

        Args:
            image: Input image (numpy array or PIL Image)
            confidence_threshold: Optional confidence threshold override

        Returns:
            Tuple containing:
            - annotated_image: Image with detection boxes drawn
            - detections: DataFrame with detection results
        """
        if self.model is None:
            self.model = self.load_model()
            if self.model is None:
                return None, None

        # Set confidence threshold if provided
        if confidence_threshold is not None:
            self.model.conf = confidence_threshold

        # Run inference
        results = self.model(image)

        try:
            # Try torch.hub format first
            detections = results.pandas().xyxy[0]
            annotated_image = np.array(results.render()[0])
        except:
            try:
                # Try yolov5 package format
                detections = results.pandas().xyxy[0]
                annotated_image = np.array(results.render()[0])
            except:
                # Manual processing
                detections = []
                annotated_image = np.array(image)
                
                for *box, conf, cls in results.xyxy[0].cpu().numpy():
                    if conf >= self.model.conf:
                        x1, y1, x2, y2 = box
                        class_name = self.model.names[int(cls)]
                        
                        detections.append({
                            'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                            'confidence': conf, 'class': int(cls),
                            'name': class_name
                        })
                
                detections = pd.DataFrame(detections)
                
                # Draw boxes using utils function
                from .utils import draw_boxes
                annotated_image = draw_boxes(np.array(image), detections)

        return annotated_image, detections
