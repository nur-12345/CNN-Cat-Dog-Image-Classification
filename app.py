import os
import glob
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="CNN Image Classification", layout="wide")
st.title("🐱🐶 CNN Image Classification App")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
epochs = st.sidebar.slider("Training Epochs", 1, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=0)

# Dataset path
image_dir = st.sidebar.text_input(
    "Dataset Path",
    "/Users/nupurshivani/Downloads/cnn_image_classification/Cat&Dog_ImgClassification/image"
)
default_img_size = (150, 150)

if not os.path.exists(image_dir):
    st.error("Dataset path does not exist. Please provide a valid folder path.")
    st.stop()

# --------------------------
# Load dataset
# --------------------------
@st.cache_data
def load_data(image_dir, img_size):
    X, y = [], []
    class_names = ["cat", "dog"]  # fixed labels since only 2 classes

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))

    if not image_paths:
        return np.array([]), np.array([]), class_names

    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)

            # Assign label from filename
            fname = os.path.basename(image_path).lower()
            if "cat" in fname:
                y.append(0)
            elif "dog" in fname:
                y.append(1)
            else:
                continue  # skip files without "cat" or "dog" in name
        except:
            continue

    if not X:
        return np.array([]), np.array([]), class_names

    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return np.array(X, dtype="float32") / 255.0, np.array(y), class_names


X, y, class_names = load_data(image_dir, default_img_size)

if len(X) == 0:
    st.error("⚠️ No images loaded. Please ensure filenames contain 'cat' or 'dog'.")
    st.stop()

st.write(f"✅ Loaded {len(X)} images.")

# Show sample images
st.subheader("🖼️ Sample Images")
cols = st.columns(5)
for i in range(min(5, len(X))):
    cols[i].image(X[i], caption=class_names[y[i]], use_container_width=True)
st.markdown("---")

# --------------------------
# Train Model
# --------------------------
if st.button("🚀 Train Model"):
    st.info("Starting model training...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(default_img_size[0], default_img_size[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    with st.spinner("Training in progress..."):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, verbose=1)

    # Plot accuracy & loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['accuracy'], label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title('Model Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.legend()

    st.pyplot(fig)

    st.success("🎉 Training Complete!")
    model.save("cnn_model.h5")
    st.info("✅ Model saved as cnn_model.h5")
st.markdown("---")

# --------------------------
# Prediction
# --------------------------
st.subheader("🔮 Predict on New Image")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if os.path.exists("cnn_model.h5"):
        model = load_model("cnn_model.h5")
        input_shape = model.input_shape[1:3]

        img_resized = cv2.resize(img, input_shape)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0) / 255.0

        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Predicting..."):
            probs = model.predict(img_array, verbose=0)[0]

        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]

        st.markdown(f"### ✅ Prediction: **{class_names[pred_idx]}**")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.error("⚠️ Train the model first before making predictions.")
