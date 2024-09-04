# Pose Detection Library

`pose_detection` is a Python library that allows you to create and detect human poses using MediaPipe. This library enables you to create pose models from both real-time video and pre-captured images, validate those poses with high accuracy, and detect them during live video streams.

## Features
- Create pose models from real-time webcam input.
- Create pose models from datasets containing images.
- Detect and validate poses using a simple API.
- High performance and accuracy using MediaPipe.
- Flexible pose validation system to compare pose models with detected poses in real-time.

## Installation

You can install the `pose_detection` library directly from PyPI using `pip`:

```bash
pip install pose-detection
```

Alternatively, you can clone the repository and install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Real-Time Pose Model Creation

You can create pose models using your webcam by following these steps:

```python
from pose_detection.pose_model_creator import PoseModelCreator

creator = PoseModelCreator()
creator.real_time_pose_creation()  # This will prompt you to create poses using your webcam
```

### 2. Pose Detection from Live Camera

After creating the pose models, you can detect those poses in real-time from your webcam:

```python
import cv2
from pose_detection.pose_detector import PoseDetector

detector = PoseDetector("pose_models.json")  # Load the JSON file with pose models
cap = cv2.VideoCapture(0)

previous_pose = None  # Track the previously detected pose

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_points = detector.process_frame(frame)
    pose_name, confidence = detector.validate_pose(detected_points)

    # Only print when a new pose is detected
    if pose_name and pose_name != previous_pose:
        print(f"Pose: {pose_name}, Confidence: {confidence}%")
        previous_pose = pose_name

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. Create Pose Models from Image Dataset

To create pose models from an image dataset, use the `create_pose_from_images()` method. Ensure that your dataset is structured as shown below.

```python
from pose_detection.pose_model_creator import PoseModelCreator

creator = PoseModelCreator()

# Specify the path to your dataset
dataset_path = 'DATASET'

# Create pose models from the dataset
creator.create_pose_from_images(dataset_path)
```

## Dataset Structure

To create pose models from images, your dataset should be organized as follows:

```
DATASET/
│
├── HAND_UP/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
├── HAND_DOWN/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
└── WAVE/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg
```

### Explanation:
- **`DATASET/`**: The main folder that contains all pose categories.
- **Pose folders**: Each pose has its own folder (e.g., `HAND_UP/`, `HAND_DOWN/`), and each folder contains multiple images demonstrating that pose.

After creating the dataset, you can generate pose models with the following code:

```python
creator.create_pose_from_images('DATASET')
```

This will generate the pose models and save them into a JSON file named `pose_models.json`.

## Pose Model JSON Format

The generated pose model file is in JSON format and contains an array of poses. Each entry in the array consists of a `message` representing the pose name and a `pose_model` containing the pose validation data.

```json
[
    {
        "message": "HAND_UP",
        "pose_model": [...]
    },
    {
        "message": "HAND_DOWN",
        "pose_model": [...]
    }
]
```

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe

You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

## Contributing

Feel free to submit issues or pull requests for improvements!

---

This README contains everything a user would need to understand and utilize the `pose_detection` library, from installation to dataset setup and usage examples.