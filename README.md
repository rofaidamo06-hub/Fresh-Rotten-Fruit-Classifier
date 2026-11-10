# 🍎 Fresh vs Rotten Fruit Classifier

A machine learning-based computer vision application designed to classify fruits and vegetables into Fresh or Rotten categories to assist in food quality detection and smart inventory management.

---

## Features
- Image Preprocessing & Feature Extraction: Utilizes OpenCV and Color Histograms to extract numerical features from images.
- Machine Learning Models: Trained using classical ML algorithms (SVM and Decision Tree).
- Interactive Web App: Built with Streamlit for real-time user predictions and smooth deployment.

---

## Tech Stack
- Python
- Scikit-Learn (for SVM & Decision Tree models)
- OpenCV & NumPy (for image processing)
- Streamlit (for web deployment)

---

## 📂 Project Structure
```text
fresh/
│
├── dataset/             # Training and testing datasets
├── app.py               # Streamlit web application script
├── svm.py               # SVM model training script
├── tree.py              # Decision Tree model training script
├── modelS.pkl           # Saved SVM model (Pickle)
├── modelT.pkl           # Saved Decision Tree model (Pickle)
└── requirements.txt     # Required Python libraries
