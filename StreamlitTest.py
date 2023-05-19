import streamlit as st
import json
from pathlib import Path
from typing import Any, Union
import os
import numpy as np
import cv2
import requests
import os
from PIL import Image
import io


# nut endpoint id (classification): f1afca98-96a0-44da-8cda-8a14c1ddf65b
# rail endpoint id (object detection): bcd2a492-71ac-4778-9cd1-572260127866


def resize(img, size):
    image = Image.open(img)
    image.thumbnail((size, size))
    return image

def inferDir(filename: Union[Path, str], endpoint_id) -> Any:
    """
    Run inference on an image
    :param filename: path to the image file
    :return: dictionary object representing the inference results
    """
    headers = {
        'apikey':  'txfcbpqmuzht1yd2mfiz7vx46x0w4un',
        'apisecret': 'ankmh8tih6bmahuwgdeft7725zlif5wasdbkpz1bedl9x58oxy55ygs2ah7e1v',
    }

    params = {
        'endpoint_id':  endpoint_id,
    }
    img = Image.open(filename)
    output = io.BytesIO()
    img.save(output, format='JPEG')
    files = {
        'file': output.getvalue()
    }

    response = requests.post('https://predict.app.landing.ai/inference/v1/predict', 
                             params=params, headers=headers, files=files)
    if(response.status_code == "429"):
        st.write("Too many requests to server (HTTP Status Code 429)")
    return json.loads(response.text)

def inferUpload(UploadedFile, endpoint_id):
    """
    Run inference on an image
    :param filename: path to the image file
    :return: dictionary object representing the inference results
    """
    headers = {
        'apikey':  'txfcbpqmuzht1yd2mfiz7vx46x0w4un',
        'apisecret': 'ankmh8tih6bmahuwgdeft7725zlif5wasdbkpz1bedl9x58oxy55ygs2ah7e1v',
    }

    params = {
        'endpoint_id':  endpoint_id,
    }

    files = {
        'file': UploadedFile.getvalue()
    }

    response = requests.post('https://predict.app.landing.ai/inference/v1/predict',
                            params=params, headers=headers, files=files)
    if(response.status_code == "429"):
        st.write("Too many requests to server (HTTP Status Code 429)")
    print(response.status_code)
    try:
        return json.loads(response.text)
    except json.decoder.JSONDecodeError:
        inferUpload(UploadedFile, endpoint_id)


def main():

    
    model_type = st.selectbox("Choose which model type you would like", 
                 ("Classification", "Object Detection", "Segmentation"))

    model_num = st.text_input("Input the endpoint id of your model")

    img_size = st.slider("Image size", min_value = 100, max_value = 800, value = 300, step = 100)

    if (model_type == "Object Detection"): 
        width = st.slider("Min bounding box width", min_value = 0, max_value = 800, value = 0, step = 10)
        height = st.slider("Min bounding box height", min_value = 0, max_value = 800, value = 0, step = 10)

    st.session_state['endpoint_id'] = model_num

    picture_upload = st.file_uploader("Click here to upload an image (or multiple)", accept_multiple_files=True)

    demo = st.checkbox("Would you like to see a demo?")    

    if (picture_upload is not None and model_type == "Classification"):
        if isinstance(picture_upload, list):
            for pic in picture_upload:
                inference = inferUpload(pic, st.session_state['endpoint_id'])
                st.write(pic.name + ":\n")
                st.image(resize(pic, img_size))
                st.write("prediction: " + inference["predictions"]["labelName"])    

    if (picture_upload is not None and model_type == "Object Detection"):
        if isinstance(picture_upload, list):
            for pic in picture_upload:
                inference = inferUpload(pic, st.session_state['endpoint_id'])
                img = cv2.imdecode(np.fromstring(pic.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
                st.write(pic.name + ":\n")
                num_bounding_box = 0
                cv2.imwrite("Output Images/output_" + pic.name, img)
                for pred in inference["backbonepredictions"]: # Loop through each prediction
                    
                    result = img.copy()
                    # Define dimensions for bounding box
                    x1 = inference["backbonepredictions"][pred]["coordinates"]["xmin"]
                    y1 = inference["backbonepredictions"][pred]["coordinates"]["ymin"]
                    x2 = inference["backbonepredictions"][pred]["coordinates"]["xmax"]
                    y2 = inference["backbonepredictions"][pred]["coordinates"]["ymax"]

                    # Output bounding box
                    if x2-x1 >= width and y2-y1 >= height:
                        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.imwrite("Output Images/output_" + pic.name, result)
                        img = cv2.imread("Output Images/output_" + pic.name)
                        num_bounding_box += 1
                st.image(resize("Output Images/output_" + pic.name, img_size))  
                st.write("Number of bounding boxes: " + str(num_bounding_box))



    # Demo models
    if demo:

        # Classification Demo Model
        if (model_type == "Classification"):
            nut_types = ["brazil", "chestnut", "almond"]
            for nut in nut_types:
                for i in range(200):
                    filename = nut + str(i) + ".JPG"
                    if(os.path.isfile("Demo Images/" + filename)):
                        inference = inferDir("Demo Images/" + filename, 'f1afca98-96a0-44da-8cda-8a14c1ddf65b')
                        st.write(filename + ":\n")
                        st.image(resize("Demo Images/" + filename, img_size))
                        st.write("prediction: " + inference["predictions"]["labelName"])


        # Object Detection Demo Model
        if (model_type == "Object Detection"):
            for i in range(10008, 127173):
                filename = str(i) + ".JPG"
                if(os.path.isfile("Demo Images/" + filename)):
                    inference = inferDir("Demo Images/" + filename,'bcd2a492-71ac-4778-9cd1-572260127866')
                    img = cv2.imread("Demo Images/" + filename)
                    st.write(filename + ":\n")
                    num_bounding_box = 0
                    cv2.imwrite("Output Images/output_" + filename, img)
                    for pred in inference["backbonepredictions"]: # Loop through each prediction
                        result = img.copy()

                        # Define dimensions for bounding box
                        x1 = inference["backbonepredictions"][pred]["coordinates"]["xmin"]
                        y1 = inference["backbonepredictions"][pred]["coordinates"]["ymin"]
                        x2 = inference["backbonepredictions"][pred]["coordinates"]["xmax"]
                        y2 = inference["backbonepredictions"][pred]["coordinates"]["ymax"]

                        # Output bounding box
                        if x2-x1 >= width and y2-y1 >= height:
                            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.imwrite("Output Images/output_" + filename, result)
                            img = cv2.imread("Output Images/output_" + filename)
                            num_bounding_box += 1
                    st.image(resize("Output Images/output_" + filename, img_size))  
                    st.write("Number of bounding boxes: " + str(num_bounding_box))

if __name__ == "__main__":
    main() 
