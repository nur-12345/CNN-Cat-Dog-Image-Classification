
🐱🐶 CNN Cat & Dog Image Classification

A Deep Learning project built using Convolutional Neural Networks (CNNs) to classify images of cats and dogs.
The project comes with an interactive Streamlit frontend, a training pipeline, and a demo script for quick experimentation.

📌 Features

    🖼️ Image Classification between cats and dogs using CNN.
    ⚡ Streamlit Web App (app.py) for training, visualization, and predictions.
    📊 Training History Plots for accuracy and loss.
    🔮 Safe Prediction Mode with confidence percentage.
    🛠️ Easy-to-run training pipeline (main.py) and demo script (demo.py).
    
📂 Project Structure

    ├── app.py             # Streamlit web application (UI for training & prediction)
    ├── main.py            # CNN model training script
    ├── demo.py            # Quick demo to load & test dataset
    ├── requirements.txt   # Required dependencies
    └── image/           # Cat & Dog images (provide your own dataset path)
⚙️ Installation & Setup

1. Clone the repository
  
       git clone https://github.com/your-username/cnn-cat-dog-classifier.git
       cd cnn-cat-dog-classifier

4. Create virtual environment (optional but recommended)
   
       python3 -m venv venv
       source venv/bin/activate     # Mac/Linux
       venv\Scripts\activate        # Windows
5. Install dependencies

       pip install -r requirements.txt

🚀 Usage

1. Run the Streamlit App
   
       streamlit run app.py
   
 Set dataset path in the sidebar.
 Adjust training parameters (epochs, batch size, learning rate).
 Train the model interactively.
 Upload an image to test predictions.

3. Train from CLI (without UI)

       python main.py

4. Run quick demo
   
       python demo.py

📊 Model Architecture

    Conv2D layers for feature extraction.
    MaxPooling2D layers for downsampling.
    Dense + Dropout layers for classification.
    Final activation: sigmoid (binary classification) / softmax (multiclass).

🎯 Sample Output

    Training Progress
    Accuracy & Loss plots are generated after training.
    
Prediction Example
  Prediction: 🐱 Cat  
  Confidence: 96.4%

📌 Requirements
  All dependencies are listed in requirements.txt.
  
Key packages:

    TensorFlow / Keras
    NumPy, OpenCV
    scikit-learn
    Streamlit
    Matplotlib

👨‍💻 Author
 Nupur Shivani
