import cv2
from src.models.player_detection import detect_players
from src.models.goal_detection import detect_goal, load_goal_detector, transforms
from src.utils.visualization import draw_players, draw_goal

def process_video(video_path, save_path=None):
    """
    Process a video to detect players and goals.

    Parameters:
    - video_path (str): Path to the input video file.
    - save_path (str, optional): Path to save the processed video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Load goal detection model and transformations
    goal_model = load_goal_detector(model_path='models/goal_detector.pth')
    goal_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if save_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players
        players = detect_players(frame)
        
        # Detect goal
        goal_detected = detect_goal(frame, goal_model, goal_transform)
        
        # Draw players and goal on frame
        frame = draw_players(frame, players)
        frame = draw_goal(frame, goal_detected)
        
        if out:
            out.write(frame)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()