
# Face Swapper using InsightFace

This repository contains code for a real-time face swapping application using the InsightFace library. It allows you to swap faces in live video streams captured from a camera.

## Prerequisites

- Python 3.9
- OpenCV (cv2)
- InsightFace (Python library for face analysis)

## Installation

1. Clone the repository:

   
   git clone https://github.com/Dat28060596/Face_swapper
   

2. Install the required Python packages:

   
   pip install -r requirements.txt
   

## Usage

1. Connect your camera to your computer.

2. Modify and run the `main.py` script:

   ```bash
   python main.py
   ```

3. Press 'q' to exit the application.

## Configuration

- **Camera Selection**: If you have multiple cameras connected to your computer, you can change the camera index in the script (`camera = cv2.VideoCapture(0)`).

- **Face Image for Swapping**: You can specify the path to the face image you want to use for swapping by providing the file path to `face_want_to_swap`. Ensure that the image contains a clear frontal face.

- **Model**: This script uses pre-trained models for face detection and face swapping. You can replace them with other models as needed.

## Credits

This project is built using the [InsightFace](https://github.com/deepinsight/insightface) library.

## License

[MIT License](LICENSE)

Feel free to use and modify this code according to your needs.
