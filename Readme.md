# Automatic Number Plate Recognition (ANPR) using YOLOv3 and EasyOCR

This project aims to demonstrate Automatic Number Plate Recognition (ANPR) using YOLOv3 object detection for detecting license plates and EasyOCR for optical character recognition (OCR) to extract text from the detected license plates. 

## Requirements
- Python 3.10.8
- OpenCV
- NumPy
- EasyOCR
- YOLOv3 model weights
- YOLOv3 model configuration file
- YOLOv3 class names file

You can install Python dependencies via pip:

pip install opencv-python numpy easyocr

## Usage
Clone this repository to your local machine:

git clone https://github.com/your-username/your-repo.git
cd your-repo

Download the YOLOv3 model weights and configuration file and place them in the appropriate directory. You can download the YOLOv3 model weights from here.

Place your input images in a directory.

Run the script main.py to start api:


python main.py

The script will detect license plates in the input images, perform OCR on the detected license plates, and print the recognized text along with confidence scores.


