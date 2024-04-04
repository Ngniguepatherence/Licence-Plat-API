import cv2
import numpy as np
import streamlit as st
from PIL import Image
import easyocr

reader = easyocr.Reader(['fr','en'])


# funtion to convert the image into grayscale
def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# define funtion to detect edges
def detect_edges(img):
    edges = cv2.Canny(img,100,200)
    return edges

# define function to detect faces

# def detect_faces(img):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface-default.xml')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1,4)
#     for (x, y,w,h) in faces:
#         cv2.rectangle(img,(x,y), (x+w, y+h), (255,0,0), 6)
#     return img

def Extract_data(img):
    reader = easyocr.Reader(['en','fr'])
    result = reader.readtext(img)
    for (bbox,text,prob) in result:
        print(f'Text: {text}, Probability: {prob}')
    return result

def extract_licence_plate_number(img):
    plate_cascade = cv2.CascadeClassifier('/Users/THERENCE/haarcascade_license_plate_rus_16stages.xml')

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect licence plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,minSize=(100,100))
    
    plate_numbers = []
    for (x,y,w,h) in plates:
        # crop the detected licences plate region
        plate_img = img[y:y+h, x:x+w]

        # Extract text from licence plate image
        plate_text = reader.readtext(plate_img)

        # If text is detected, append it to the list of plate number
        if plate_text:
            plate_numbers.append(plate_text[0][1])
    return plate_numbers



# set the title of the web app
st.title("Opencv App")

# add a button to upload the image file from user
uploaded_file = st.file_uploader("choose an image.../video...", type=["jpg","png","jpeg","mp4"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension.lower() in ["jpg", "png", "jpeg"]:

        # convert the file to an opencv image
        file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        # file_bytes = uploaded_file.getva;
        img = cv2.imdecode(file_bytes, 1)

        # Display the original image
        st.image(img, channels="BGR", use_column_width=True)

        # When the grayscale button is clicked, convert the image to grayscale
        if st.button('Convert to grayscale'):
            img_gray = convert_to_gray(img)
            st.image(img_gray, use_column_width=True)

        # When the Detect Edges button is clicked, detect edges in the image
        if st.button("Detect Edges"):
            img_edges = detect_edges(img)
            st.image(img_edges, use_column_width=True)

        # When the Detect Faces button is clicked, detect faces in. the image
        if st.button('Extract data'):
            data = Extract_data(img)
            for (bbox,text,prob) in data:
                st.text(f'Text: {text}, Probability: {prob}')
        if st.button('licence plate number'):
            datas = extract_licence_plate_number(img)
            st.text(data)
    elif file_extension.lower() == "mp4":
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
    