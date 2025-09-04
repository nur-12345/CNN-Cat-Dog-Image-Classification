
import os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set paths
image_dir = '/Users/nupurshivani/Downloads/cnn_image_classification/Cat&Dog_ImgClassification/image'

# Check if dataset exists
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Dataset folder not found: {image_dir}")
else:
    print(f"Dataset folder found: {image_dir}")



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
        img = cv2.resize(img, img_size)
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
            
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue

# Convert to numpy arrays and normalize
X = np.array(X, dtype="float32") / 255.0
#[241/255, 245/255, 246/255] ≈ [0.945, 0.961, 0.965]
y = np.array(y)

print(f"Loaded {len(X)} images with shape {X.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=batch_size,
    verbose=1
)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Function to predict single image
def predict_image(image_path, model, target_size=(100, 100)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return
    
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0) / 255.0
    
    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    plt.imshow(img)
    plt.title(f"Prediction: {label} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.show()

# Test prediction
test_image_path = os.path.join(image_dir, 'cat.18.jpg')  # Change to your test image
predict_image(test_image_path, model)