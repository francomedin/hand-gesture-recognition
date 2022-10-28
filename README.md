


# Hand Gesture Recognition
## _Project to recognize hand signals_


Hand Gesture Recognition is a project to create signals and the recognize them in streaming or video.

## Features

- Create your own signals.
- Train and test the model.
- Monitor your model´s perfomance.
- Predict from streaming or video.mp4 files.

## Project tree

.
 * example_data
   * A
        * images
   * B
        * images
   * C
        * images
   * D
        * images
 * model
   * keras_model.h5
   * labels.txt
 * .gitignore
 * Classifier.py
 * data_collection.py
 * detect_hand.py
 * evaluation.py
 * README.md
 * requirements.txt
 * test_hand_label.py
 * utils.py
 * video_from_file.py




## Tech


- Python
- OpenCV
- Keras
- Scikit-learn


## Installation

pip install -r requirements.txt

- For left-right-both hand detection:
*       python detect_hand.py
- For hand and gesture detection from web cam:
*       python test_hand_label.py
- For hand and gesture detection from video (.mp4):
*       python video_form_file.py


## Model Result


|Labels|  precision |  recall |  f1-score |  support |
|---|---|---|---|---|
|0|1.0|0.62|0.76|60|
|1|1.0|0.95|0.97|60|
|2|0.70|0.65|0.68|75|
|3|0.67|1.0|0.81|64|



|Metrics|  Values |  Values |Values |  support |
|---|---|---|---|---|
|accuracy|-|-|0.8|259|
|macro avg|0.84|0.80|0.80|259|
|weighted avg|0.83|0.80|0.80|259|

For info about multiclass metrics https://bit.ly/3gQmBjE


## Next steps

- Develop a CNN to improve perfomance.
- Segment hands of the image to remove the background.
- Create new signals (just 4 at the moment)
- Investigate edge devices to run the model.
- Pass variables through console (argparser)