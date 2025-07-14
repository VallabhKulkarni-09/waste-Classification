# Waste Classification

## Overview
This project is a machine learning-based waste classification system. It uses a trained model (`best.pt`) to identify and classify different types of waste from images, helping improve recycling and waste management processes.

## Features
- Image-based waste classification
- Pre-trained model included
- Easy-to-use Python application

## Setup

### Option 1: Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/VallabhKulkarni-09/waste-Classification.git
   cd waste-Classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Docker
1. Clone the repository:
   ```bash
   git clone https://github.com/VallabhKulkarni-09/waste-Classification.git
   cd waste-Classification
   ```

2. Build the Docker image:
   ```bash
   docker build -t waste-classification .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 8501:8501 waste-classification
   ```

4. Open your web browser and navigate to `http://localhost:8501`

Note: The Docker version includes all dependencies and is ready to run without additional setup.

## Usage
Run the application:
```bash
python app.py
```
Follow the prompts to classify waste images.


## How it Works
This application uses a YOLOv5 model to classify waste as dry or wet. The app is built with Streamlit and provides an interactive interface for users. Key features include:

## Example Results
Below are sample images showing the app's detection results and interface:

### Dry Waste Detection
![Dry Waste Example 1](examples/dry%201.png)
![Dry Waste Example 2](examples/dry%202.png)

### Wet Waste Detection
![Wet Waste Example 1](examples/wet%201.png)
![Wet Waste Example 2](examples/wet%202.png)

### Application UI
![Streamlit UI](examples/streamlit%20UI.png)

- **Input Methods:**
  - Upload an image file
  - Use your device camera
  - Upload a video file for frame-by-frame detection
- **Detection:**
  - The model analyzes the input and detects waste objects, drawing bounding boxes and labels.
  - Confidence scores are displayed for each detection.
- **Results:**
  - The app shows the detected waste type (dry or wet) with confidence.
  - Disposal recommendations are provided for each waste type.
  - All detections are listed with their confidence scores.

### Example Workflow
1. Start the app with `python app.py`.
2. Select your preferred input method in the sidebar.
3. Upload an image, capture a photo, or upload a video.
4. View detection results, confidence scores, and disposal recommendations.

## Model
The model file `best.pt` is a PyTorch model trained for waste classification. You can retrain or fine-tune it as needed.

## Model Performance
The following metrics summarize the performance of the YOLOv5 model on the validation set:

| Metric    | Meaning                                                                 |
|----------|-------------------------------------------------------------------------|
| P        | Precision: Proportion of predicted objects that are correct              |
| R        | Recall: Proportion of actual objects that were correctly detected        |
| mAP50    | Mean Average Precision at IoU 0.5: Detection/localization at 50% overlap|
| mAP50-95 | Mean Average Precision averaged over IoU 0.5 to 0.95                    |

**Overall (all classes):**
- Precision: 0.751
- Recall: 0.931
- mAP50: 0.843
- mAP50-95: 0.539

**Per Class:**
- Dry: P=0.581, R=0.91, mAP50=0.721, mAP50-95=0.499
- Wet: P=0.922, R=0.951, mAP50=0.965, mAP50-95=0.58

## Dataset
This project uses the [Dry and Wet Waste dataset](https://universe.roboflow.com/bhushan-kinge/dry-and-wet-waste) from Roboflow for training the YOLOv5 model. The dataset contains labeled images of dry and wet waste, suitable for object detection tasks.

## Contact
For questions or support, contact [Vallabh Kulkarni](mailto:vallabhkulkarni028@gmail.com).

## Contributing
Contributions are welcome! If you would like to improve this project, please follow these steps:

1. Fork the repository and create your branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. Test your changes to ensure they work as expected.
4. Submit a pull request with a detailed description of your changes.

### Guidelines
- Follow PEP8 style for Python code.
- Add docstrings and comments for new functions or modules.
- If adding new features, update the documentation and README accordingly.
- For bug fixes, describe the issue and how your fix resolves it.

## Docker Development
To modify the Docker setup:

1. Edit the `Dockerfile` to update build steps or dependencies
2. Rebuild the image after changes:
   ```bash
   docker build -t waste-classification .
   ```
3. Run with volume mount for development:
   ```bash
   docker run -p 8501:8501 -v $(pwd):/app waste-classification
   ```
