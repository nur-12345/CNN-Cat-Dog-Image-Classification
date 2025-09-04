import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set paths
image_dir = ' /Users/nupurshivani/Downloads/cnn_image_classification/Cat&Dog_ImgClassification/image'
img_size = (100, 100)
batch_size = 32

# Get all image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_paths)

# Load images and labels
X = []
y = []

for image_path in image_paths:
    try:
        # Load and preprocess image

        img = cv2.imread(image_path)
        print(img)
        print(img.shape)
     
        img = cv2.resize(img,img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
        
        # Get label from filename==
        filename = os.path.basename(image_path)
        if filename.startswith('cat'):
            y.append(0)  # 0 for cat
        elif filename.startswith('dog'):
            y.append(1)  # 1 for dog
        else:
            print(f"Unknown file: {filename}, skipping")
            X.pop()  # Remove the appended image if label is invalid
            continue
        break
            
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue
print(X)
print(X[0].shape)
print(y[0])

#[241/255, 245/255, 246/255] ≈ [0.945, 0.961, 0.965]