import cv2
import mediapipe as mp
from Classifier import Classifier
from utils import draw_rectangle, predict, detect_hand

# Media pipe initiation
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Variables
classifier = Classifier(
    "model/converted_keras/keras_model.h5", "model/converted_keras/labels.txt"
)
index = 0
OFFSET = 25
LABELS = ["A", "B", "C", "D"]


# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
    max_num_hands=2,
) as hands:

    while cap.isOpened():

        success, image = cap.read()
        h, w, c = image.shape

        # Convert from BGR to RGB, process and then go back.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # 
        if results.multi_hand_landmarks:

            try:
                # Get the left or right hand
                image_hand, label = detect_hand(image, hands)
                # Draw the bounding box
                x_max, x_min, y_max, y_min, image = draw_rectangle(results, image=image)
                img_crop = image[
                    y_min - OFFSET : y_max + OFFSET, x_min - OFFSET : x_max + OFFSET
                ]
                # Get the prediction and index label
                prediction, index, img_white = predict(img_crop, h, w, classifier)

                # Only show predictions for the right hand.
                if label == "Right":
                    cv2.putText(
                        image,
                        LABELS[index],
                        (x_max, y_max - 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (255, 0, 255),
                        2,
                    )
                # cv2.imshow("Image-cropped", img_crop)
                # cv2.imshow('Image-cropped', image_hand)
                cv2.imshow("Image-white", img_white)

                key = cv2.waitKey(1)

            except:
                pass

        cv2.imshow("Image", image)
        
        # Exit video
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
