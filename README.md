# Electronics Object Detection

## Overview
This project focuses on the application of AI techniques for object detection, specifically targeting electronic items. Utilizing a custom dataset and state-of-the-art machine learning algorithms, the project aims to accurately identify and locate electronic objects within images.

## Project Structure
- `config.yaml` - Configuration file for model and training parameters.
- `dataset.yaml` - Dataset definition and path configurations.
- `load_openimages_dataset.py` - Script to download and prepare the Open Images dataset for training.
- `object_detection.py` - Core script for implementing object detection models.
- `predict.py` - Script for making predictions on new images.
- `train.py` - Script for training the object detection model.
- `runs/` - Directory for storing model runs and outputs.
- `tests/` - Directory containing test scripts and validation checks.

## Features
- Customizable model training and prediction scripts.
- Support for various object detection algorithms.
- Integration with the Open Images dataset for training.
- Automated dataset preparation and loading.

## Getting Started

### Prerequisites
- Python 3.6 or later
- PyTorch
- OpenCV
- Other dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
```bash
git clone https://github.com/smk75/Electronics_Object_Detection-main.git
```
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Usage
1. Prepare the dataset by running:
```bash
python load_openimages_dataset.py
```
2. Train the model using:
```bash
python train.py
```
3. Make predictions on new images by running:
```bash
python predict.py --image_path path/to/your/image.jpg
```

## Contributing
Contributions are welcome and greatly appreciated. If you have suggestions for improving this project, please fork the repo and create a pull request, or open an issue with the tag "enhancement".

## License
This project is licensed under the MIT License - see the LICENSE file for details.
