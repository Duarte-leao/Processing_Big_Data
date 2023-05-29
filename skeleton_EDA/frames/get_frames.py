import cv2
import os

def extract_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return

    # Get the directory of the video file
    video_dir = os.path.dirname(video_path)

    # Initialize variables
    frame_count = 0

    # Read the video frames
    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # Break the loop if no frame was retrieved
        if not ret:
            break

        # Save the frame with the frame number as the filename
        frame_filename = os.path.join(video_dir, f"{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    video.release()

    print(f"Frames extracted: {frame_count}")
# Specify the video file name
video_file = "girosmallveryslow2.mp4"

# Call the function to extract frames
extract_frames(video_file)
