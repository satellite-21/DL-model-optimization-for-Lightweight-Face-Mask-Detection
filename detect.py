import cv2
import mediapipe as mp
import time
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils.image_utils import img_to_array
import numpy as np
from keras.models import load_model
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
import multiprocessing
import tensorflow as tf

cap = cv2.VideoCapture(0)
interpreter = tf.lite.Interpreter(model_path="mobilenet1.tflite", num_threads=multiprocessing.cpu_count())
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
	while cap.isOpened():
		success, image = cap.read()
		image = cv2.resize(image,(640,480))
		start=time.time()
		#Convert the BGR image to RGB.
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = face_detection.process(image)
		# crop face from results of mediapipe

		# Draw the face detection annotations on the image.
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		if results.detections:
			for id, detection in enumerate(results.detections):
				bboxC = detection.location_data.relative_bounding_box
				h, w, c = image.shape
				boundBox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
				face = image[boundBox[1]:boundBox[1]+boundBox[3], boundBox[0]:boundBox[0]+boundBox[2]]
				face = cv2.resize(face, (128, 128))
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = preprocess_input(np.array(face))
				face = np.expand_dims(face, axis=0)
				face = np.array(face, dtype="float32")
				interpreter.resize_tensor_input(input_details[0]['index'], (1, 128, 128, 3))
				interpreter.resize_tensor_input(output_details[0]['index'], (1, 2))
				interpreter.allocate_tensors()
				input_details = interpreter.get_input_details()
				output_details = interpreter.get_output_details()
				interpreter.set_tensor(input_details[0]['index'], face)
				interpreter.invoke()
				preds = interpreter.get_tensor(output_details[0]['index'])
				label = "Mask" if preds[0][0] >  preds[0][1] else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(preds[0][0], preds[0][1]) * 100)

		# display the label and bounding box rectangle on the output
		# frame
				cv2.rectangle(image, boundBox, color, 2)
				cv2.putText(image, label, (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

		
		end = time.time()
		totalTime = end-start

		fps = 1/totalTime
		print(fps)

		cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

		cv2.imshow('Face Detection', image)
		if cv2.waitKey(5) & 0xFF == 27:
			break