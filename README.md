---
title: BinGENIUS – Smart Waste Classifier
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# ♻️ BinGENIUS – AI-Powered Waste Classification

**BinGENIUS** is an AI-powered waste classification application that uses Deep Learning and Computer Vision to automatically identify and categorize waste into **Organic** and **Recyclable** categories.

Built using a VGG16-based Convolutional Neural Network (CNN), the application helps users identify the appropriate waste disposal category simply by uploading an image.

## 🚀 Features

- **AI-Powered Classification:** Identifies organic and recyclable waste using a trained CNN model.
- **Image Upload:** Upload waste images for instant classification.
- **Real-Time Prediction:** Get predictions through an interactive Gradio interface.
- **Confidence Scores:** View the model's prediction confidence.
- **Disposal Recommendations:** Receive suggested waste-bin categories and disposal tips.
- **User-Friendly Interface:** Simple and accessible interface for waste identification.

## 🧠 Model Architecture

The project uses a **VGG16-based Convolutional Neural Network** with additional custom classification layers.

| Parameter | Configuration |
|---|---|
| Model | VGG16 + Custom CNN |
| Input Shape | 224 × 224 × 3 |
| Output Layer | Dense (1, Sigmoid) |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Classification | Binary Classification |
| Framework | TensorFlow / Keras |

### Waste Categories

| Category | Description |
|---|---|
| 🍎 Organic | Biodegradable waste such as food scraps and vegetable waste |
| ♻️ Recyclable | Recyclable materials such as paper, plastic, and metal |

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model Architecture:** VGG16, CNN
- **Image Processing:** NumPy, Pillow
- **Interface:** Gradio
- **Deployment:** Hugging Face Spaces

## 📂 Project Structure

```text
BinGENIUS/
│
├── app.py              # Gradio application
├── model.h5            # Trained CNN model
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/bin-genius
cd bin-genius
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The application will start locally and provide a Gradio interface.

## 📸 How to Use

1. Open the BinGENIUS application.
2. Upload an image of waste or capture one using your webcam.
3. Click **Identify Waste**.
4. View the predicted category and confidence score.
5. Follow the suggested disposal recommendations.

## 🌍 Project Objective

BinGENIUS aims to demonstrate how Artificial Intelligence and Deep Learning can contribute to smarter waste identification and responsible waste management.

By simplifying waste classification, the project promotes awareness of waste segregation and encourages environmentally responsible practices.

## ⚠️ Disclaimer

This application provides AI-based classification guidance. Actual waste disposal and recycling practices may vary depending on local waste-management regulations.

---

**Developed with Python, TensorFlow, CNN and Gradio.**

♻️ *Making waste classification smarter, one image at a time.*