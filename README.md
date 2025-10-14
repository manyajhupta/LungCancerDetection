# 🫁 Lung Cancer Detection using CNN

This application uses a **Convolutional Neural Network (CNN)** to detect lung cancer from DICOM scan data. It includes a user-friendly **Tkinter-based GUI** for importing data, preprocessing, training the model, and displaying results, including predictions and a confusion matrix.

---

## 📦 Features

- GUI interface built using Tkinter  
- Data import from DICOM scans and label CSV  
- Preprocessing of CT scan slices (resizing & averaging)  
- 3D CNN model for cancer classification  
- Displays accuracy, confusion matrix, and patient-wise prediction  
- Interactive buttons for step-by-step control  

---

## 📁 Folder Structure

```
project/
├── Images/
│   └── Lung-Cancer-Detection.jpg
├── sample_images/
│   └── <Patient_ID>/
│       └── *.dcm
├── stage1_labels.csv
├── imageDataNew-10-10-5.npy
├── lung_cancer_detection.py
└── README.md
```

---

## 🛠️ Requirements

Install required packages using:

```bash
pip install numpy pandas pydicom matplotlib opencv-python tflearn tensorflow==1.15
```

> ⚠️ Note: This project uses **TensorFlow 1.x** via `tf.compat.v1` for compatibility with TFLearn.

---

## 🚀 How to Run

1. Place patient DICOM folders inside `sample_images/`  
2. Ensure `stage1_labels.csv` is present in the project folder  
3. Run the application:

```bash
python lung_cancer_detection.py
```

---

## 🧠 CNN Model Overview

- Input shape: (10×10×5)
- Architecture:
  - 5× 3D Convolution + MaxPooling layers
  - Fully Connected Layer + Dropout
  - Final Softmax for binary classification

---

## 📊 Output

- Final accuracy on validation set  
- Confusion matrix display  
- Table of predicted vs actual results per patient

---

## 📌 Data Set

Have taken 50 patients as a sample dataset for training and validation. Link is available below:

Sample Dataset Images: https://qnm8.sharepoint.com/:f:/g/Ep5GUq573mVHnE3PJavB738Bevue4plkiXyNkYfxHI-a-A?e=UVMWne

Sample Dataset CSV for above images: CSV File

---

## 👨‍⚕️ Objective

To assist in **early detection** of lung cancer, reduce false positives, and support radiologists in faster diagnosis using deep learning.

 
