import cv2
import matplotlib.pyplot as plt

def draw_players(frame, players):
    """
    Draw detected players on a frame.

    Parameters:
    - frame (np.ndarray): Input frame.
    - players (list of tuples): List of detected player bounding boxes.

    Returns:
    - frame (np.ndarray): Frame with drawn bounding boxes.
    """
    for (startX, startY, endX, endY) in players:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame

def draw_goal(frame, goal_detected):
    """
    Draw goal detection result on a frame.

    Parameters:
    - frame (np.ndarray): Input frame.
    - goal_detected (bool): True if a goal is detected, False otherwise.

    Returns:
    - frame (np.ndarray): Frame with goal detection result.
    """
    if goal_detected:
        cv2.putText(frame, "Goal Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def display_frame(frame):
    """
    Display a frame using OpenCV.

    Parameters:
    - frame (np.ndarray): Frame to display.
    """
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()