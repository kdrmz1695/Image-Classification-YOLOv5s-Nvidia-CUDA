import cv2
from video_processing import process_frame
from model import load_model
from utils import time_it
import time

@time_it
def process_video_sequential(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, model)

        fps = frame_count / (time.time() - start_time)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sequential Processing (CPU)", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"Sequential FPS (CPU): {fps:.2f}")
    print(f"Sequential Processing Time: {total_time:.2f} seconds")
    print(f"Number of processed frames: {frame_count}")

if __name__ == "__main__":
    video_path = "people3.mp4"
    model = load_model("cpu")
    process_video_sequential(video_path, model)
