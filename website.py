import numpy as np
import cv2
import os
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise', 5: 'disgust', 6: 'fear'}
# load json and create model
json_file = open('expression_facial_vgg2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("expression_facial_vgg2.h5")

#load face
try:
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{
"urls": [
"turn:13.250.13.83:3478?transport=udp"
],
"username": "YzYNCouZM1mhqhmseWk6",
"credential": "YzYNCouZM1mhqhmseWk6"
}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    
    st.header("Facial Emotion Detections")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    


if __name__ == "__main__":
    main()
