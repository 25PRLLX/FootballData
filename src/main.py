import os
from src.data.load_data import load_video
from src.utils.video_processing import process_video

def main():
    video_path = 'data/videos/match1.mp4'
    save_path = 'data/videos/output_match1.mp4'
    
    cap = load_video(video_path)
    
    process_video(cap, save_path=save_path)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()