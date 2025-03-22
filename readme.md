# ASL Classification

This project focuses on classifying American Sign Language (ASL) gestures using a machine learning model. The work is implemented in the `classification_model.ipynb` notebook.

## Project Overview

The goal of this project is to build a model that can accurately classify ASL gestures from images. The main steps involved in this project are:

1. **Data Collection**: Gathering a dataset of images representing different ASL gestures.
2. **Data Preprocessing**: Preparing the images for training by resizing, normalizing, and augmenting the data.
3. **Model Building**: Creating a convolutional neural network (CNN) to classify the ASL gestures.
4. **Model Training**: Training the CNN on the preprocessed dataset.
5. **Model Evaluation**: Evaluating the performance of the model on a test dataset.
6. **Model Deployment**: Saving the trained model for future use.

## Files

- `classification_model.ipynb`: The Jupyter notebook containing the code for data preprocessing, model building, training, and evaluation.

## Dependencies

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Open `classification_model.ipynb` in Jupyter Notebook.
4. Run the notebook cells to preprocess the data, train the model, and evaluate its performance.

## Results

The model achieved an accuracy of 98% on the test dataset. Further improvements can be made by fine-tuning the model and using a larger dataset.

## Future Work

- Experiment with different model architectures.
- Collect more data to improve model accuracy.
- Implement real-time gesture recognition.

## Acknowledgements

- Data sourced from Kaggle ASL alphabet dataset.
- [Any other acknowledgements]
