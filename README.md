# Smart-Waste-Sorting-Assistant
Building AI course project
# Summary:

An AI-powered mobile app that helps people sort their waste correctly by taking a photo of an item and identifying whether it belongs to plastic, metal, paper, glass, or organic waste. This reduces recycling errors and promotes sustainable living.
# Background:

The Problem:
People often throw recyclable materials into the wrong bins, causing contamination and reducing recycling efficiency.
Complex items (e.g., coated cups, mixed plastics) confuse people, leading to landfill waste.
#Why It’s Important:

Proper sorting reduces landfill waste and improves recycling.

Supports environmental sustainability and global climate goals.

# My Motivation:

I want to use AI for environmental good and encourage eco-friendly habits in my community.

# How is it used?

User takes a photo of an item with a mobile app.

The AI model predicts the waste category (plastic, metal, paper, glass, organic).

The app gives sorting instructions, e.g.,

“This is recyclable paper, put it in the blue bin.”

“This is organic waste, compost it if possible.”

Schools and municipalities can use it for education and public awareness.

# Code Snippet :

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

# Load pre-trained MobileNet model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def predict_waste(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x.reshape((1, 224, 224, 3)))
    preds = model.predict(x)
    print("Predicted class:", decode_predictions(preds, top=1)[0])

predict_waste("plastic_bottle.jpg")

# Data Sources and AI Methods

#Data Sources:
TrashNet Dataset (https://github.com/garythung/trashnet) – an open-source dataset of waste images.

Additional local images collected from recycling centers.

# AI Methods:
Convolutional Neural Networks (CNNs) for image classification.

Transfer Learning with pre-trained models (MobileNet, ResNet) for faster training and higher accuracy.

# Challenges:
Misclassification of items made of multiple materials (e.g., plastic-coated paper).

Lighting conditions: Low-light or blurry photos may reduce accuracy.

Data limitation: Requires large, high-quality datasets.

# What next?:
Add barcode scanning to identify products automatically.

Integrate with city recycling systems to provide real-time bin locations.

Develop a smart AI-powered bin that auto-sorts waste.

# Acknowledgments:
Inspired by sustainability goals and open-source projects.

Dataset: TrashNet Dataset / MIT License.

Open-source AI libraries: TensorFlow, PyTorch.




