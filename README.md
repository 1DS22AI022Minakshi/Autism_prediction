

# Autism_prediction– Streamlit App Using CNN

This Streamlit application predicts the likelihood of Autism Spectrum Disorder (ASD) from image input using a Convolutional Neural Network (CNN) model.

## 🧠 Project Overview

This project is designed to assist in early detection of ASD through image-based predictions. It leverages deep learning techniques, particularly CNNs, to analyze facial features and provide prediction scores.

## 🚀 Features

- Upload an image to receive an ASD likelihood prediction
- Real-time inference using a trained CNN model
- User-friendly interface built with Streamlit
- Visualization of model predictions and probabilities

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Model**: Convolutional Neural Network (CNN)
- **Libraries**: TensorFlow / Keras, OpenCV, NumPy, Matplotlib

## 🧪 Model Details

The CNN model was trained on a curated dataset of facial images, labeled for ASD detection. The model achieves high accuracy in identifying potential ASD-related features.

## 📦 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/1DS22AI022Minakshi/Autism_prediction
    cd asd-predictor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

## 📁 File Structure

```
├── app.py               # Main Streamlit application
├── model/               # Trained CNN model
├── utils.py             # Image processing utilities
├── requirements.txt     # Required Python libraries
└── README.md            # Project documentation
```

## 📌 Disclaimer

This tool is intended for educational and research purposes only. It should not be used for medical diagnosis or treatment decisions.

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)

## 🧑‍💻 Author

Developed by Minakshi

