import cv2
import insightface

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 for default camera, you can change it if you have multiple cameras

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize FaceAnalysis and swapper models
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx') #https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view

# Load the face image to swap
face_want_to_swap = cv2.imread(r"C:\Users\Admin\PycharmProjects\Swap_face\amber-heard-4k-2018-l4-3840x2400.jpg")

# Continuously capture frames
while True:
    # Capture a frame
    ret, frame = camera.read()

    if not ret:
        print("Error: Could not capture frame.")
        break

    # Get faces from the frame
    faces = app.get(frame)

    # Get source face from the image to swap
    swap_face = app.get(face_want_to_swap)
    source_face = swap_face[0]

    # Swap faces in the frame
    for face in faces:
        frame = swapper.get(frame, face, source_face, paste_back=True)

    # Display the frame
    cv2.imshow('Captured Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
