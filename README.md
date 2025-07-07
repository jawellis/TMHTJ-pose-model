# Jerk Step Trainer 

## Overview

This project is divided into two main parts:

1. **The User Application**  
   An interactive web app that helps users learn the "jerk" dance move using their webcam, pose recognition with MedaPipe , and direct visual and textual feedback.
2. **The Training Tool**  
   A tool to record your own pose samples, save them, and use them for model training.

---

## Main Features

- **MediaPipe Pose Detection**: Real-time pose detection via webcam.
- **KNN Classifier**: Recognizes four different poses based on trained examples.
- **Step-by-step Feedback**: Guides the user with instructions, visual feedback, and progress indicators.
- **Custom Data Training**: Easily record and save your own samples in JSON format.
- **Model Evaluation & Accuracy**: Test and evaluate the accuracy of your model and visualize results with a confusion matrix.

---

## Installation & Getting Started

### Requirements

- Node.js & npm 
- Webcam

### Installation

1. **Clone the project**
   git clone <TMHTJ-pose-model>

2.**Dependencies**
cd app
npm install
cd ../train
npm install
3. **User app**
cd app
npm start
4. **Traiing tool**
cd train
npm start


