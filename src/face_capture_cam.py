import cv2
from mtcnn.mtcnn import MTCNN
from datetime import datetime
import os
import time

custom_thresholds = [0.7, 0.7, 0.8]  
# Load MTCNN for face detection
mtcnn_detector = MTCNN(steps_threshold = custom_thresholds)

# Create a directory to save the captured images

IMG_PATH = './DataSet/FaceData/raw/'
count = 5
usr_name = input("Input ur name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)

os.makedirs(USR_PATH, exist_ok=True)

# Open a video capture object for the webcam (camera index 0)
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# Counter for captured images
capture_count = 0
# Set the frame rate
fps = 1  # 1 frame per second
frame_interval = int(video_capture.get(5) / fps)  # 5 is the index for frame rate
last_capture_time = time.time()


while capture_count < count:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to RGB (MTCNN expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = mtcnn_detector.detect_faces(rgb_frame)

    # Draw rectangles around detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

        # Save the captured face image every second
        current_time = time.time()
        if current_time - last_capture_time >= 1:
            capture_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            image_name = f"{capture_time}-{capture_count}.jpg"
            image_path = os.path.join(USR_PATH, image_name)
            captured_face = frame[y:y+height, x:x+width]
            cv2.imwrite(image_path, captured_face)
            last_capture_time = current_time
            capture_count += 1
            cv2.putText(frame, f"Image Index: {capture_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    # Display the resulting frame
    cv2.imshow('Face Capture from Webcam', frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()


