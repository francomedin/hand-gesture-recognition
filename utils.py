from turtle import width
import cv2
import numpy as np
import math
import mediapipe as mp
from google.protobuf.json_format import MessageToDict





def draw_rectangle(result_mediapipe, image):
    x_max = 0
    y_max = 0
    h,w,c = image.shape
    x_min = w
    y_min = h 
    offset = 25


    for hand in result_mediapipe.multi_hand_landmarks:
       
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

    cv2.rectangle(image, (x_min- offset,y_min - offset),(x_max + offset, y_max+ offset), (0,255,0),2)

    
    return x_max, x_min, y_max, y_min, image




def predict(img_crop, h, w, classifier = None):
    image_size = 300

    img_white = np.ones((image_size,image_size,3), np.uint8) * 255
    aspect_ratio = h/w
    prediction, idx = '', 0

    if aspect_ratio > 1:

        k = image_size / h
        width_calculated = math.ceil(w * k)
        img_resized = cv2.resize(img_crop, (width_calculated, image_size))
        # Center the image
        width_gap = math.ceil(image_size - width_calculated/2)
        img_white[:, width_gap:width_calculated + width_gap] =  img_resized
        if classifier:
            prediction, idx = classifier.getPrediction(img_white, draw=False)
        

    else:
        k = image_size / w
        height_calculated = math.ceil(k * h) # ceil round up
        img_resized = cv2.resize(img_crop, (image_size, height_calculated))
        # Center the image
        height_gap = math.ceil((image_size - height_calculated)/2)
        img_white[height_gap: height_calculated + height_gap, :] =  img_resized
        if classifier:
            prediction, idx = classifier.getPrediction(img_white,draw=False)

    if classifier:
        return prediction, idx, img_white
    else:
        return img_white




def detect_hand(image, hands):
    # Initializing the Model
    
    label = ''
    # Flip the image(frame)
    image = cv2.flip(image, 1)

	# Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Process the RGB image
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

		# Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
                # Display 'Both Hands' on the image
            cv2.putText(image, 'Both Hands', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 255, 0), 2)

        # If any hand present
        else:

            for i in results.multi_handedness:
            
                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']
                
                print(label)
                
                if label == 'Left':
                
                    # Display 'Left Hand' on
                    # left side of window
                    cv2.putText(image, label+' Hand',
                                (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

                if label == 'Right':
                    
                    # Display 'Left Hand'
                    # on left side of window
                    cv2.putText(image, label+' Hand', (460, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
    return image, label

        