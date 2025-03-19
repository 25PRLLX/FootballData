import torch
from ultralytics.yolo.engine.model import YOLO

def detect_players(frame, model_path='runs/train/exp/weights/best.pt', conf_threshold=0.5):
    """
    Detect players in a frame using a pre-trained YOLOv5 model.

    Parameters:
    - frame (np.ndarray): Input frame.
    - model_path (str): Path to the YOLOv5 model weights.
    - conf_threshold (float): Confidence threshold for detections.

    Returns:
    - players (list of tuples): List of detected player bounding boxes.
    """
    model = YOLO(model_path)
    
    results = model(frame, conf=conf_threshold)
    
    players = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            players.append((r[0], r[1], r[2], r[3]))
    
    return players