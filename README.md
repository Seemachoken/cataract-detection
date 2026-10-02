---
title: Cataract Detection
sdk: docker
app_port: 7860
pinned: false
---

# Cataract Detection

Research prototype for classifying uploaded fundus images with a CNN and MobileNetV2 ensemble.

The app loads `cnn_model.h5` and `mobilenet_model.h5` at startup. Store these model files with Git LFS when pushing this Space. Training datasets and notebooks are not needed to run the web app.

This project is for academic research only and is not a medical diagnosis tool. Users should consult a qualified eye-care professional.
