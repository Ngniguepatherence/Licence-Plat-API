import cv2
import numpy as np
import streamlit as st
from PIL import Image
import easyocr
import os

reader = easyocr.Reader(['fr','en'])

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

        output = reader.readtext(license_plate_gray)

        for out in output:
            text_bbox, text, text_score = out
            print(text, text_score)  # Modify this line as needed

        return license_plate_gray


def extract_licence_plate_number_video():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, you can change it if needed

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('Press Space to Capture', frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        # If space key is pressed, capture the frame
        if key == ord(' '):
            # Release the camera
            cap.release()
            cv2.destroyAllWindows()

            # Load class names
            with open('./model/class.names', 'r') as f:
                class_names = [j[:-1] for j in f.readlines() if len(j) > 2]

            # Load YOLO model
            net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

            # Get image dimensions
            H, W, _ = frame.shape

            # Convert image
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True)

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

                license_plate = frame[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

                frame = cv2.rectangle(frame,
                                    (int(xc - (w / 2)), int(yc - (h / 2))),
                                    (int(xc + (w / 2)), int(yc + (h / 2))),
                                    (0, 255, 0),
                                    15)

                license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

                _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                output = reader.readtext(license_plate_gray)

                for out in output:
                    text_bbox, text, text_score = out
                    print(text, text_score)  # Modify this line as needed

                return license_plate_gray

        # If 'q' key is pressed, exit
        elif key == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()



# set the title of the web app
st.title("Opencv App")

# add a button to upload the image file from user
uploaded_file = st.file_uploader("choose an image.../video...", type=["jpg","png","jpeg","mp4","gif"])
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
            img_edges = extract_licence_plate_number_video()
            st.image(img_edges, use_column_width=True)

        # When the Detect Faces button is clicked, detect faces in. the image
        if st.button('Extract data'):
            data = Extract_data(img)
            for (bbox,text,prob) in data:
                st.text(f'Text: {text}, Probability: {prob}')
        if st.button('licence plate number'):
            img_data = extract_licence_plate_number(img)
            st.image(img_data, use_column_width=True)
    elif file_extension.lower() == "mp4":
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
    