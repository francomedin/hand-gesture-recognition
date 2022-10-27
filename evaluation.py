from sklearn.metrics import classification_report
import cv2
from Classifier import Classifier
import os

# Load the model
model = Classifier(
    "model/converted_keras/keras_model.h5", "model/converted_keras/labels.txt"
)
labels = ["A", "B", "C", "D"]


def evaluation(data_folder: str):
    """Predict all the labels in the test folder"""

    counter = 0
    diferences = 0
    lst_bad_inference = []
    y_true = []
    y_pred = []
    for root, _, files in os.walk(data_folder, topdown=False):

        for name in files:
            if root.split("/")[1].split("\\")[0] == "test":
                image_path = os.path.join(root, name)
                image = cv2.imread(image_path)

                _, idx = model.getPrediction(image)
                counter += 1
                label = root.split("\\")[1]
                y_true.append(labels.index(label))
                y_pred.append(idx)

                print(f"Label {label} | indice: {labels[idx]}")
                if label != labels[idx]:
                    diferences += 1
                    lst_bad_inference.append({"Prediction": _, "y_true": label})

    print(diferences)
    print(f"Total images: {counter}")
    for inference in lst_bad_inference:
        print(
            f'Bad inference: predicted: {labels[idx]} | y_true = {inference["y_true"]}'
        )

    print(classification_report(y_true, y_pred))

    return y_pred, y_true


if __name__ == "__main__":
    evaluation("data/")
