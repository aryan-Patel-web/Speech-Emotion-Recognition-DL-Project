# 🎤 Speech Emotion Recognition (Flask)

This project is a machine learning web app that predicts emotions (like happy, sad, or angry) from `.wav` audio clips. It uses MFCC, Chroma, and Mel features to analyze voice patterns, and is trained on the RAVDESS dataset using an MLPClassifier.

---

## 🚀 Features

- Upload `.wav` files to predict emotions like happy, sad, angry, calm, etc.
- Audio feature extraction using `librosa`: MFCC, Chroma, and Mel spectrograms
- Model trained using `MLPClassifier` from scikit-learn
- Flask-powered web app with a clean HTML+CSS frontend

---

## 🧠 Model Details

- Dataset: [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- Features: MFCC, Chroma, Mel
- Classifier: MLPClassifier (Multi-layer Perceptron)
- Accuracy: ~72%
- Saved Model: `model.pkl` (Pickle format)

---

## 📁 Project Structure

```
Speech-Emotion-Recognition-Flask/
├── app.py
├── model.pkl
├── extract_feature.py
├── requirements.txt
├── templates/
│   └── index.html
```

---

## ▶️ Running the App Locally

### 1. Clone the repo

```bash
git clone https://github.com/aryan-Patel-web/Speech-Emotion-Recognition-Flask.git
cd Speech-Emotion-Recognition-Flask
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you face a `resampy` error:
```bash
pip install resampy
```

### 3. Run the Flask app

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 🧪 Try with Sample `.wav` Files

You can record your own or use audio from the RAVDESS dataset. Make sure files are mono and 16kHz for best results.

---

## 📦 Requirements

- Flask  
- numpy  
- scikit-learn  
- librosa  
- resampy  

Install all with:

```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Aryan Patel**  
A machine learning enthusiast passionate about combining AI and sound.


