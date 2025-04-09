# 🚦 Traffic Light & Sign Recognition System using AI and Computer Vision

This project implements a real-time traffic light and sign recognition system for autonomous vehicles using **OpenCV** and **TensorFlow**. It detects and classifies traffic lights (Red, Yellow, Green) and traffic signs (e.g., Stop, Speed Limits) through a combination of classical computer vision techniques and convolutional neural networks (CNNs).

---


## 👩‍💻 Team Members

- Azra Selvitop 
- Yiğit Temiz
  
## 🧠 Project Summary

This system:
- Detects **traffic signs** using color segmentation and geometric shape recognition.
- Classifies signs using a custom-trained CNN on the **GTSRB** dataset.
- Detects **traffic lights** and classifies them using a separate CNN trained on the **LISA Traffic Light Dataset**.

It can process both **static images** and **video streams**.

## 📌 Features

- 🚦 Traffic Light Classification: Red, Yellow, Green
- 🛑 Traffic Sign Classification: 43 classes (e.g., stop, speed limits, pedestrian crossing)
- 🧪 Augmented training data
- 📸 Real-time prediction with OpenCV
- 💾 Dataset preprocessing, annotation visualization, and shape detection

---

## 📚 Datasets Used

- [GTSRB Dataset (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/gtsrb_news.html)
- [LISA Traffic Light Dataset](https://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html)

---


## Train or Load Models
You can either use our pre-trained models or retrain from scratch:

- Sign model: sign_recognition/train_sign_cnn.py

- Light model: light_recognition/train_light_cnn.py
