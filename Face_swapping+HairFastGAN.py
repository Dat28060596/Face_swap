import cv2
import insightface
import time
from pathlib import Path
from hair_swap import HairFast, get_parser
import torchvision.transforms as T
import torch
import numpy as np
# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 for default camera, you can change it if you have multiple cameras

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize FaceAnalysis and swapper models for face swapping
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 ensures GPU usage if available
swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

# Load the face image to swap
face_want_to_swap = cv2.imread(r"C:\Users\Admin\OneDrive\Pictures\Screenshots\Screenshot 2024-07-12 172841.png")
swap_face = app.get(face_want_to_swap)
source_face = swap_face[0]

# Initialize HairFast model for hair swapping with GPU support
model_args = get_parser()
hair_fast = HairFast(model_args.parse_args([]))

# Check if a GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hair_fast = hair_fast.to(device)

# Initialize FPS calculation variables
fps = 0
frame_count = 0
start_time = time.time()

# Continuously capture frames
while True:
    # Capture a frame
    ret, frame = camera.read()

    if not ret:
        print("Error: Could not capture frame.")
        break

    # Step 1: Perform face swapping
    faces = app.get(frame)
    for face in faces:
        frame = swapper.get(frame, face, source_face, paste_back=True)

    # Step 2: Perform hair swapping
    input_dir = Path("/content/HairFastGAN/input")
    face_path = input_dir / 'face.png'
    shape_path = input_dir / 'hair_style.png'
    color_path = input_dir / 'color.png'

    # Move paths to the GPU if applicable
    face_path = face_path.to(device)
    shape_path = shape_path.to(device)
    color_path = color_path.to(device)

    final_image, face_align, shape_align, color_align = hair_fast.swap(face_path, shape_path, color_path, align=True)

    # Move final result back to CPU for further processing (if needed)
    final_image = final_image.cpu()

    # Convert the result back to an image format suitable for OpenCV
    final_image = T.functional.to_pil_image(final_image)
    final_image = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)

    # Calculate FPS
    frame_count += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = end_time

    # Display the FPS on the frame
    cv2.putText(final_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Captured Frame', final_image)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
