# Skeleton Overlay
import cv2
import numpy as np
from pathlib import Path

# Define the connections between keypoints (adjust indices based on your data's keypoint order)
# Example for 6 keypoints (modify according to your skeleton structure)
CONNECTIONS = [
    #(0, 1),            # 0-4 keypoints for face
    #(0, 2),
    #(0, 3),
    #(0, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (11, 13),
    (13, 15),
    (11, 12),
    (6, 12),
    (12, 14),
    (14, 16)
]

# Confidence threshold (ignore keypoints below this value)
CONF_THRESHOLD = 0.5

def parse_skeleton_data(file_path):
    """Read skeleton data from a text file."""
    skeleton_data = []
    with open(file_path, 'r') as f:
        for line in f:
            numbers = list(map(float, line.strip().split()))
            keypoints = []
            for i in range(0, len(numbers), 3):
                x, y, conf = numbers[i], numbers[i+1], numbers[i+2]
                if conf >= CONF_THRESHOLD:
                    keypoints.append((int(x), int(y)))
                else:
                    keypoints.append(None)  # Mark low-confidence points
            skeleton_data.append(keypoints)
    return skeleton_data

def draw_skeleton(frame, keypoints):
    """Draw keypoints and connections on a frame."""
    # Draw keypoints
    for idx, kp in enumerate(keypoints):
        if kp is not None and idx >= 5:  # 0-4 keypoints for face, >=5 keypoints for face removed
            x, y = kp
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red circles
    
    # Draw connections
    for (i, j) in CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            kp1 = keypoints[i]
            kp2 = keypoints[j]
            if kp1 is not None and kp2 is not None:
                cv2.line(frame, kp1, kp2, (0, 255, 0), 2)  # Green lines
    return frame

def main():
    folder_input_vid = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/Camera/Skeleton_Videos/"
    filenames_vid = [str(f) for f in Path(folder_input_vid).rglob('*') if f.is_file()]
    folder_input_ske = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/Camera/Skeleton_Data/"
    filenames_ske = [str(f) for f in Path(folder_input_ske).rglob('*') if f.is_file()]
    folder_output_vid = "C:/Users/kubil/Documents/STUDIUM/Master/4_Masterarbeit_Code/Camera/Overlay/"
    filenames_vid_out = [f.stem for f in Path(folder_input_vid).rglob('*') if f.is_file()]
    
    for i in range(len(filenames_vid)):
        # Load skeleton data
        skeleton_data = parse_skeleton_data(filenames_ske[i])

        # Open video
        cap = cv2.VideoCapture(filenames_vid[i])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(folder_output_vid + filenames_vid_out[i] + "_overlay.mp4", fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay skeleton if data exists for this frame
            if frame_idx < len(skeleton_data):
                keypoints = skeleton_data[frame_idx]
                frame = draw_skeleton(frame, keypoints)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print("Done! Output saved to:", folder_output_vid + filenames_vid_out[i] + "_overlay.mp4")

if __name__ == "__main__":
    main()