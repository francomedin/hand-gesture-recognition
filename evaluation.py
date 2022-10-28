from sklearn.metrics import classification_report
import cv2
from Classifier import Classifier
import os

# Load the model
model = Classifier(
    "model/converted_keras/keras_model.h5", "model/converted_keras/labels.txt"
)
LABELS = ["A", "B", "C", "D"]


def evaluation(data_folder: str):
    """
    Walk between the data_folder and classify the images. Then return 
    y_pred and y_true/y_test.
    ----------

    data_folder : str

    Returns
    -------
    y_pred: list
    y_true: list
    """
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

                # Get predictions and save labels
                _, idx = model.getPrediction(image)
                counter += 1
                label = root.split("\\")[1]
                y_true.append(LABELS.index(label))
                y_pred.append(idx)

                print(f"Label {label} | indice: {LABELS[idx]}")
                
                # Save wrong predictions
                if label != LABELS[idx]:
                    diferences += 1
                    lst_bad_inference.append({"Prediction": _, "y_true": label})

    print(diferences)
    print(f"Total images: {counter}")
    for inference in lst_bad_inference:
        print(
            f'Bad inference: predicted: {LABELS[idx]} | y_true = {inference["y_true"]}'
        )

    print(classification_report(y_true, y_pred))

    return y_true, y_pred


if __name__ == "__main__":
    evaluation("data/")
