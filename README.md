# Real-Time Emotion Detection

![Project Logo](path/to/logo.png) <!-- Replace with your project logo/image -->

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
Real-Time Emotion Detection is a project aimed at recognizing human emotions through facial expressions using deep learning techniques. The model leverages Convolutional Neural Networks (CNNs) to classify emotions in real-time, making it applicable in various domains such as mental health monitoring, user experience enhancement, and interactive systems.

## Features
- Real-time emotion detection from webcam input
- Supports multiple emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- High accuracy based on the trained model
- User-friendly interface for easy interaction

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pallav110/Real-Time-Emotion-Detection.git
   
2.cd Real-Time-Emotion-Detection

3.python -m venv .venv

4.Activate the virtual environment:
  On Windows:
.venv\Scripts\activate
On macOS/Linux:
source .venv/bin/activate

5.pip install -r requirements.txt

6.To run the real-time emotion detection, execute the following command:
python test_webcam.py



Model Architecture
The project utilizes a Convolutional Neural Network (CNN) designed specifically for emotion detection. Below is a brief overview of the architecture:

Input Layer: Accepts images resized to 48x48 pixels.
Convolutional Layers: Feature extraction through multiple convolutional layers followed by ReLU activations.
Fully Connected Layers: Classifies emotions based on the extracted features.

Dataset
The model is trained on the FER2013 dataset, which consists of labeled facial expressions in grayscale. More details can be found here.

Results
The model achieves the following accuracy on the test set:

Overall Accuracy: 46.84%
Accuracy for each emotion:
Angry: 26.51%
Disgust: 60.36%
Fear: 16.99%
Happy: 76.32%
Neutral: 47.36%
Sad: 27.19%
Surprise: 71.00%


License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any inquiries or suggestions, feel free to reach out:

Name: Pallav Sharma
GitHub: pallav110

