import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
import math
from utils import predict


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

offset = 25
image_size = 300

folder = "data/A"
counter = 0

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        h,w,c = image.shape

        # Shift from BGR TO RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Get the results of hand processing.
        results = hands.process(image)
        image.flags.writeable = True
        # Convert back from RGB To BGR to show the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        if results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            
            for num, hand in enumerate(results.multi_hand_landmarks):
                
                # Draw the rectangle
                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x 
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

                # Draw rectangle and landmarks
                cv2.rectangle(image, (x_min- offset,y_min - offset),(x_max + offset, y_max+ offset), (0,255,0),2)
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS) # (image, hand coorninates, landmarks conections)

            try:
                # Create a white image for background
                img_crop = image[y_min - offset:y_max + offset,x_min - offset:x_max + offset]
                img_white = predict(img_crop, h, w)
               

                # Show images
                cv2.imshow('Image-cropped', img_crop)
                cv2.imshow('Image-white', img_white)

                key = cv2.waitKey(1)

                # Save images with key "s"
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(os.path.join(
                        'data/C',
                        f'{uuid.uuid1()}.jpg'
                        ),
                        img_white)
                    print(f'{counter} images saved')

            except:
                pass

        # Detections
        cv2.imshow('Image', image)
        
        if cv2.waitKey(10) & 0XFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()