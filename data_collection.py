import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
import math



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
    # Tracking confidence after hand detection
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        #(480, 640, 3)
        h,w,c = image.shape
        # Flip image horizontally
        # image= cv2.flip(image,1)

        # Shift from BGR TO RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
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
                # Create a white image
                img_crop = image[y_min - offset:y_max + offset,x_min - offset:x_max + offset]
                img_white = np.ones((image_size,image_size,3), np.uint8) *255

                aspect_ratio = h/w
                

                if aspect_ratio > 1:
                    k = image_size / h
                    width_calculated = math.ceil(k * w) # ceil round up
                    img_resized = cv2.resize(img_crop, (width_calculated, image_size))
                    # Center the image
                    width_gap = math.ceil(image_size - width_calculated/2)
                    img_white[:, width_gap:width_calculated + width_gap] =  img_resized
                else:
                    k = image_size / w
                    height_calculated = math.ceil(k * h) # ceil round up
                    img_resized = cv2.resize(img_crop, (image_size, height_calculated))
                    # Center the image
                    height_gap = math.ceil((image_size - height_calculated)/2)
                    img_white[height_gap: height_calculated + height_gap, :] =  img_resized

                cv2.imshow('Image-cropped', img_crop)
                cv2.imshow('Image-white', img_white)
                key = cv2.waitKey(1)

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