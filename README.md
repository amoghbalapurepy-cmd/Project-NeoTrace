# Project-NeoTrace

> **Observe. Understand. Interact.**

## 📖 About

**Project-NeoTrace** is an experimental real-time computer vision and augmented reality system that combines object detection, persistent tracking, gesture recognition, facial analysis, and AR interactions into a single intelligent platform.

This project aims to transform a standard camera into an interactive visual assistant capable of understanding and responding to its environment in real time.

---

## ⚠️ Development Status

**Project-NeoTrace is currently in its early development stage.**

This repository represents:

* 🎓 **My first high-end computer vision project**
* 🧪 An experimental draft and learning project
* 🚧 A work-in-progress that still requires optimization, improvements, and additional features

Some systems may be incomplete, contain bugs, or require further refinement. The project is continuously evolving as I learn and improve my skills in computer vision and AI development.

Contributions, suggestions, and feedback are always appreciated.

---

# ✨ Features

## 🎯 Real-Time Object Detection

* YOLOv8-powered object recognition
* Detection of 80+ common object classes
* Confidence-based filtering

## 📌 Persistent Object Tracking

* Unique IDs for tracked objects
* IoU-based tracking system
* Bounding-box smoothing
* Multi-frame confirmation to reduce false positives

## ✋ Hand Tracking & Gesture Recognition

* 21-point hand landmark detection
* Recognizes gestures such as:

  * Open Palm
  * Fist
  * Pointer
  * Peace Sign
  * L-Shape
  * Thumbs Up
  * And more...

## 😊 Face & Eye Tracking

* Face mesh detection
* Iris tracking
* Eye blink detection
* Facial landmark visualization

## 🏷️ Brand & Company Identification

* AI-powered brand recognition
* Zero-shot classification using CLIP
* Company information database integration

## 📱 In-Hand Object Detection

* Detects when an object is being held by the user
* Links objects and hand positions in real time

## ✨ Augmented Reality Drawing

Draw in the air using your finger and automatically generate glowing AR shapes:

* Circle
* Triangle
* Square
* Rectangle
* Pentagon
* Line

---

# 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* YOLOv8 (Ultralytics)
* NumPy
* Transformers (CLIP)
* PyTorch

---

# 📂 Installation

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

```bash
python object_identifier.py
```

---

# 🎮 Controls

| Key | Action                      |
| --- | --------------------------- |
| Q   | Quit                        |
| S   | Screenshot                  |
| H   | Toggle HUD                  |
| O   | Toggle Object Detection     |
| B   | Toggle Brand Identification |
| A   | Toggle AR Drawing           |
| C   | Clear AR Canvas             |

---

# 🎯 Vision

The long-term goal of Project-NeoTrace is to create an intelligent visual system that can:

* See
* Understand
* Track
* Interpret
* Interact with the physical world in real time.

---

# 🚀 Future Plans

* Better object tracking algorithms
* Multi-person recognition
* Voice assistant integration
* OCR and text understanding
* Scene description generation
* 3D object positioning
* Better AR interactions
* Performance optimization
* Custom-trained recognition models

---

# 📝 Note from the Developer

Project-NeoTrace started as an idea to challenge myself and learn advanced computer vision concepts.

It is my **first high-end AI/computer vision project**, and while it is still far from perfect, every line of code in this repository represents a step toward building something bigger.

Thank you for checking out this project and following its journey.

---

**Project-NeoTrace**
*Observe. Understand. Interact.*
