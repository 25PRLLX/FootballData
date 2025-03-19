import numpy as np
import cv2

def preprocess_frame(frame):
    """
    Preprocess a single frame.

    Parameters:
    - frame (np.ndarray): Input frame.

    Returns:
    - processed_frame (np.ndarray): Preprocessed frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize frame if necessary
    # gray = cv2.resize(gray, (new_width, new_height))
    
    return gray