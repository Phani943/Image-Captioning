# Image Captioning

This repository contains code and models for an Image Captioning project using deep learning. The project aims to generate captions for images using a pre-trained model.

## Project Structure

- **model/**: Contains the trained model files.
  - `image_captioner_200k.h5`: The main model file used for image captioning.

- **notebooks/**: Jupyter notebooks for data preprocessing, model training, and evaluation.
  - `download_data.ipynb`: Notebook to download and preprocess the dataset.
  - `train_model.ipynb`: Notebook to train the image captioning model.
  - `evaluate_model.ipynb`: Notebook to evaluate the model's performance.

- **pickles/**: Contains serialized objects used in the project.
  - `tokenizer.pkl`: Tokenizer used to preprocess the text data.

- **.gitattributes**: Specifies how Git LFS should handle large files.
- **README.md**: Project description and instructions.
- **model.png**: Diagram of the model architecture.
- **tokenizer_100k.json**: JSON file containing tokenizer configuration.

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
2. Install the required Python packages:

  ```sh
  Copy code
  pip install -r requirements.txt

3. Initialize Git LFS:

  ```sh
  Copy code
  git lfs install
  git lfs pull

Usage
Download Data: Use the notebooks/download_data.ipynb notebook to download and preprocess the dataset.

Train Model: Use the notebooks/train_model.ipynb notebook to train the image captioning model.

Evaluate Model: Use the notebooks/evaluate_model.ipynb notebook to evaluate the model's performance on test data.

Contributing
Contributions are welcome! Please create an issue to discuss your ideas or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
