# Image Captioning with 200K Images

This repository contains code and models for an Image Captioning project using deep learning. The project aims to generate captions for images by leveraging Convolutional layers and LSTM layers.

## Dataset

The image IDs and their captions are collected from Hugging Face, selecting the first 200k rows from 500k rows for the dataset. Using these image IDs and an ***Amazon AWS bucket***, the corresponding images were downloaded from ***Open Images***, a vast collection of diverse images. These 200k images are then placed on Kaggle. The entire data preparation process, including creating the Kaggle dataset, is detailed in the `data_preparation.ipynb` notebook.

## Project Structure

- **model/**: Contains the trained model file.
  - `image_captioner_200k.h5`: The main model file used for image captioning.

- **notebooks/**: Jupyter notebooks for data collection, model training, and evaluation.
  - `data_preparation.ipynb`: Notebook to download and preprocess the dataset.
  - `image-captioning-training.ipynb`: Notebook to train and evaluate the image captioning model.

- **pickles/**: Contains serialized objects used in the project.
  - `img_features_1.pkl`: [Link to Kaggle dataset](https://www.kaggle.com/datasets/phanichaitanya349/processed-200k)
  - `img_features_2.pkl`: [Link to Kaggle dataset](https://www.kaggle.com/datasets/phanichaitanya349/processed-200k)
  - `captions_1.pkl`: Captions of the first half of the images.
  - `captions_2.pkl`: Captions of the second half of the images.

- **tokenizer_100k.json**: TensorFlow JSON tokenizer to convert text to sequences.

- **.gitattributes**: Specifies how Git LFS should handle large files.
- **README.md**: Project description and instructions.
- **model.png**: Diagram of the model architecture.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Git
- Git LFS

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Phani943/Image-Captioning.git
   cd Image-Captioning
   ```

## Usage

### Download Data

Use the `notebooks/data_preparation.ipynb` notebook to download and preprocess the dataset. Alternatively, you can use the my dataset provided on Kaggle: [Kaggle Dataset Link](https://www.kaggle.com/datasets/phanichaitanya349/captioning-dataset-200k).

### Train Model

Use the `notebooks/image-captioning-training.ipynb` notebook to train the image captioning model. The training process includes the following steps:

1. **Data Loading:** Load the preprocessed image features and captions.

   - `img_features_1.pkl` and `img_features_2.pkl`: Contain the image features extracted using VGG16.
   - `captions_1.pkl` and `captions_2.pkl`: Contain the corresponding captions.

2. **Model Architecture:** The model consists of Convolutional layers to extract image features and LSTM layers to generate captions.

   - The image features are extracted using a pre-trained VGG16 network.
   - The captions are generated using an LSTM network, which takes the image features and the previously generated words as inputs.

3. **Training:** The model is trained using a combination of the image features and their corresponding captions.

   - The model is compiled with a categorical cross-entropy loss function and the Adam optimizer.
   - The training process includes monitoring the loss and accuracy on the validation set to prevent overfitting.
   - The model achieved 48% accuracy on both the training and validation sets.

4. **Saving the Model:** After training, the model is saved as `image_captioner_200k.h5`.

### Evaluate Model

The `notebooks/image-captioning-training.ipynb` notebook contains the code to evaluate the model's performance on test data. The evaluation includes generating captions for a set of test images and comparing them to the ground truth captions.

## Contributing

Contributions are welcome! Please create an issue to discuss your ideas or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
