import cv2
import json
import os

def load_video(video_path):
    """
    Load a video file.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    - cap (cv2.VideoCapture): Video capture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    return cap

def load_annotations(annotations_path):
    """
    Load annotations from a JSON file.

    Parameters:
    - annotations_path (str): Path to the annotations file.

    Returns:
    - annotations (dict): Loaded annotations.
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    return annotations