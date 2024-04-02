import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import easyocr
from typing import List
from pydantic import BaseModel, AnyHttpUrl

app = FastAPI()

reader = easyocr.Reader(['fr','en'])

# function to convert the image into grayscale
def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# function to detect edges
def detect_edges(img):
    edges = cv2.Canny(img, 100, 200)
    return edges

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
@app.get("/")
async def test_api():
    return {"message": "API is running successfully."}