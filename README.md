


# Hand Gesture Recognition
## _Project to recognize signals_


Hand Gesture Recognition is a project to create signals and the recognize them in streaming or video.

## Features

- Create your own signals.
- Train and test the model.
- Monitor your model´s perfomance.
- Predict from streaming or video.mp4 files.

## Folder structure

.
├── example_data           # Example images.
├── model                  # Keras model.
├── .gitignore             # Git file to ignore files.
├── classifier.py          # Model Loader.
├── data_collection.py     # Script to create labeled images with camera.
├── detect_hand.py         # Script to detect which hand is in the video. (left, right or both)
├── evaluation.py          # Script to evaluate our model based on test images.
├── README.md              # Project Description
├── requirements.txt       # Libraries and moldules used in the project.
├── test_hand_label.py     # Script to test the model with the webcam.       
├── utils.py               # Tools and utilities
└── video_from_file.md     # Script to test the model with a video.mp4



This text you see here is *actually- written in Markdown! To get a feel
for Markdown's syntax, type some text into the left window and
watch the results in the right.

## Tech

HGR uses a number of open source projects to work properly:

- Python
- OpenCV
- Keras
- Scikit-learn


## Installation

---> Crear entorno e instalar dependencias.

pip install -r requirements. txt

## Model Result


|Labels|  precision |  recall |  f1-score |  support |
|---|---|---|---|---|
|0|1.0|0.62|0.76|60|
|1|1.0|0.95|0.97|60|
|2|0.70|0.65|0.68|75|
|3|0.67|1.0|0.81|64|



---- > Corregir tabla
|accuracy|-|-|0.8|259|
|macro avg|0.84|0.80|0.80|259|
|weighted avg|0.83 |0.8|0.8|259|


## Next steps

- Develop a CNN to improve perfomance.
- Create new signals.
- Investigate edge devices to run the model.