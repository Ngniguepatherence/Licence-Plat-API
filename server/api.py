import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import uvicorn
import os
import easyocr
from typing import List
from pydantic import BaseModel, AnyHttpUrl

app = FastAPI()

# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

def NMS(boxes, class_ids, confidences, overlapThresh = 0.5):

    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)

    # Return empty lists, if no boxes given
    if len(boxes) == 0:
        return [], [], []

    x1 = boxes[:, 0] - (boxes[:, 2] / 2)  # x coordinate of the top-left corner
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)  # y coordinate of the top-left corner
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)  # x coordinate of the bottom-right corner
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)  # y coordinate of the bottom-right corner

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        # Create temporary indices
        temp_indices = indices[indices != i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
        xx2 = np.minimum(box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
        yy2 = np.minimum(box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if overlapping greater than our threshold, remove the bounding box
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

    # return only the boxes at the remaining indices
    return boxes[indices], class_ids[indices], confidences[indices]


def get_outputs(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)

    outs = [c for out in outs for c in out if c[4] > 0.1]

    return outs


def draw(bbox, img):

    xc, yc, w, h = bbox
    img = cv2.rectangle(img,
                        (xc - int(w / 2), yc - int(h / 2)),
                        (xc + int(w / 2), yc + int(h / 2)),
                        (0, 255, 0), 20)

    return img

reader = easyocr.Reader(['fr','en'])

# function to convert the image into grayscale
def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# function to detect edges
def detect_edges(img):
    edges = cv2.Canny(img, 100, 200)
    return edges

def detect_license_plates(img):
    # Load class names
    with open('./model/class.names', 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]

    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # Get image dimensions
    H, W, _ = img.shape

    # Convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # Set input to the network
    net.setInput(blob)

    # Get detections
    detections = get_outputs(net)

    # Initialize lists for bounding boxes, class ids, and scores
    bboxes = []
    class_ids = []
    scores = []

    # Process detections
    for detection in detections:
        bbox = detection[:4]
        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # Apply non-maximum suppression
    bboxes, class_ids, scores = NMS(bboxes, class_ids, scores)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en','fr'])

    # Loop through detected bounding boxes
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            15)

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(license_plate)

        for out in output:
            text_bbox, text, text_score = out
            print(text, text_score)  # Modify this line as needed

    return img  # Return the modified image with bounding boxes


# function to extract data using EasyOCR
def extract_data(img):
    result = reader.readtext(img)
    extracted_text = []
    for (bbox, text, prob) in result:
        extracted_text.append({'text': text, 'probability': prob})
    return extracted_text



# function to extract license plate numbers

class ProcessedImage(BaseModel):
    extracted_data: List

# Define FastAPI endpoints
@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the imag
    extracted_data = extract_data(img)

    processed_image = ProcessedImage(
        extracted_data=extracted_data,
    )

    return processed_image

@app.post("/licence_plate")
async def licence_plate(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the imag
    processed_image = detect_license_plates(img)

    return processed_image

@app.get("/")
async def test_api():
    return {"message": "API is running successfully."}




HOST = '0.0.0.0'

if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=8000, reload=True)