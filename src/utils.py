"""
Utility functions for waste classification application.
"""
from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def draw_boxes(image: np.ndarray, detections: pd.DataFrame) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.

    Args:
        image: Input image as numpy array
        detections: DataFrame containing detection results with columns:
                   [xmin, ymin, xmax, ymax, confidence, class, name]

    Returns:
        Annotated image with bounding boxes and labels
    """
    if detections.empty:
        return image
    
    img = image.copy()
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], 
                                  detection['xmax'], detection['ymax']])
        conf = detection['confidence']
        class_name = detection['name']
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    return img

def process_detections(detections: pd.DataFrame) -> Tuple[Optional[str], Optional[float], int]:
    """
    Process detection results to get classification results.

    Args:
        detections: DataFrame containing detection results

    Returns:
        Tuple containing:
        - class_name: Name of the detected class (str or None)
        - confidence: Confidence score (float or None)
        - num_detections: Number of detections (int)
    """
    if detections.empty:
        return None, None, 0
    
    # Get the detection with highest confidence
    best_detection = detections.loc[detections['confidence'].idxmax()]
    
    class_name = best_detection['name']
    confidence = best_detection['confidence']
    
    return class_name, confidence, len(detections)
