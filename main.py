# pip install opencv-contrib-python numpy tensorflow cvzone
import cv2 
import cvzone
import tensorflow as tf
from tensorflow.keras.preprocessing.image import *
import numpy as np


model = tf.keras.models.load_model("models/model.h5")

def detectFace(net,frame,confidence_threshold=0.8):
    frameOpencvDNN=frame.copy()
    frameHeight=frameOpencvDNN.shape[0]
    frameWidth=frameOpencvDNN.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>confidence_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,255),2,1)
    return frameOpencvDNN,faceBoxes


faceProto='models/opencv_face_detector.pbtxt'
faceModel='models/opencv_face_detector_uint8.pb'
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
faceNet=cv2.dnn.readNet(faceModel,faceProto)
padding = 5

sentiment_color = {
	"Angry":(0, 0, 255),
	"Disgust":(0, 255, 0),
	"Fear":(190, 190, 190),
	"Happy":(180, 105, 255),
	"Neutral":(0, 255, 255),
	"Sad":(100, 0, 0),
	"Surprise":(255, 0, 255)
}
def emotion_detection(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face,(48,48),interpolation=cv2.INTER_AREA)
    if np.sum([gray_face])!=0:
        roi = gray_face.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
        prediction = model.predict(roi)[0]
        label=emotion_labels[prediction.argmax()]

    return label



cap = cv2.VideoCapture(0)

while cap.isOpened():
	suc, frame = cap.read()
	if not suc:
		break
		print("unabel to get camera feed")

	res,faceBoxes=detectFace(faceNet,frame)
	if len(faceBoxes) !=0:
		for faceBox in faceBoxes:
			face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
			label = emotion_detection(face)
			print(label)
			emo = cv2.imread(f"emojies/{label}.png", cv2.IMREAD_UNCHANGED)
			emo = cv2.resize(emo, (40, 40))
			try:
				res = cvzone.overlayPNG(res, emo, [faceBox[0]+80, faceBox[1]-30])
			except:
				pass
			cv2.putText(res,f'{label}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.65,sentiment_color[label],1,cv2.LINE_AA)



	cv2.imshow("webcam", res)

	if cv2.waitKey(1)==27:
		break

