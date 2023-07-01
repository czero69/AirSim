import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def main(input_file):
    base_dir = Path(input_file).parent

    with open(input_file, 'r') as file:
        lines = file.readlines()
        
        for line in tqdm(lines, desc="Processing videos"):
            video_path = base_dir / line.strip()
            print(f"\nChecking file: {video_path}")

            if not video_path.is_file():
                print(f"File not found: {video_path}")
                continue

            dir = video_path.parent
            filename = video_path.name
            base = filename.rsplit('.', 1)[0]

            unique_id = base.split("-___-")[1]

            os.makedirs(f"{dir}/{unique_id}/screenshot", exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0

            if not height or not fps:
                print(f"Could not determine the height or the fps of the video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            pbar = tqdm(total=total_frames, desc=f"Extracting frames from {filename}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % int(10*fps) == 0:
                    if height != 2160:
                        frame = cv2.resize(frame, (int(frame.shape[1]*2160/frame.shape[0]), 2160))
                    cv2.imwrite(f"{dir}/{unique_id}/screenshot/output_{frame_count//int(10*fps):04d}.png", frame)

                frame_count += 1
                pbar.update(1)

            cap.release()
            pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video files.")
    parser.add_argument('input_file', type=str, help='Input file containing video file paths')
    args = parser.parse_args()
    
    main(args.input_file)
