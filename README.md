# 🫁 Lung Cancer Detection using 3D CNN

This application uses a **3D Convolutional Neural Network** built with modern TensorFlow/Keras to detect lung cancer from DICOM scan data. It includes a **Tkinter-based GUI** for importing data, preprocessing, training the model, and displaying results including predictions and confusion matrix.

---

## 📦 Features

- Simple GUI interface built using Tkinter  
- Import data from DICOM scans and label CSV  
- Automated preprocessing of CT scan slices (resizing & averaging)  
- 3D CNN model architecture for binary cancer classification  
- Real-time training with accuracy metrics
- Displays confusion matrix and patient-wise predictions  
- Step-by-step interactive workflow  

---

## 📁 Project Structure

```
lung-cancer-detection/
├── Images/
│   └── Lung-Cancer-Detection.jpg
├── sample_images/
│   └── <Patient_ID>/
│       └── *.dcm
├── stage1_labels.csv
├── imageDataNew-10-10-5.npy 
├── lung_cancer_detection.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lung-cancer-detection.git
cd lung-cancer-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```


---

## 🚀 How to Run

1. **Prepare your data:**
   - Place patient DICOM folders inside `sample_images/`
   - Ensure `stage1_labels.csv` is present in the project folder

2. **Run the application:**
```bash
python lung_cancer_detection.py
```

3. **Follow the GUI workflow:**
   - Click **Import Data** to load DICOM files and labels
   - Click **Pre-Process Data** to convert DICOM to numpy arrays
   - Click **Train Data** to train the model and view results

---

## 🧠 Model Architecture

The 3D CNN model processes volumetric CT scan data with the following architecture:

- **Input shape:** 10×10×5 (width × height × depth × channels)
- **Architecture:**
  - 5× 3D Convolutional layers (32 → 64 → 128 → 256 → 512 filters)
  - MaxPooling3D after each convolution
  - Flatten layer
  - Fully Connected layer (256 neurons)
  - Dropout layer (0.2 rate)
  - Output layer (2 classes - Softmax activation)

- **Training Configuration:**
  - Optimizer: Adam (learning rate: 0.001)
  - Loss: Categorical Crossentropy
  - Epochs: 100
  - Validation split: 5 patients

---

## 📊 Output

The application provides:
- **Training metrics:** Accuracy per epoch
- **Final validation accuracy:** Model performance on unseen data
- **Confusion matrix:** Visual representation of classification results
- **Patient predictions table:** Side-by-side comparison of actual vs predicted labels

---

## 📌 Sample Dataset

This project uses a sample dataset of 50 patients for demonstration:

- **Sample Dataset Images:** [SharePoint Link](https://qnm8.sharepoint.com/:f:/g/Ep5GUq573mVHnE3PJavB738Bevue4plkiXyNkYfxHI-a-A?e=UVMWne)
- **Sample Labels CSV:** [stage1_labels.csv](stage1_labels.csv)

> 💡 **Note:** This is a limited dataset for educational purposes. For production use, a larger, more diverse dataset is recommended.

---

## 🎯 Objective

To assist in **early detection** of lung cancer by:
- Reducing false positives in CT scan analysis
- Supporting radiologists with AI-powered second opinions
- Enabling faster diagnosis through automated screening
- Demonstrating practical deep learning applications in healthcare

---

## 🔧 Technical Updates

This version has been updated with:
- ✅ **TensorFlow 2.x** - Modern, actively maintained framework
- ✅ **Keras Sequential API** - Clean, intuitive model building
- ✅ **Updated dependencies** - Latest stable versions
- ✅ **PIL.Image.LANCZOS** - Replaces deprecated ANTIALIAS
- ✅ **Improved code structure** - Better naming and organization

---

## 📝 Requirements

- numpy >= 1.21.0
- pandas >= 1.3.0
- pydicom >= 2.3.0
- matplotlib >= 3.4.0
- opencv-python >= 4.5.0
- scikit-learn >= 1.0.0
- tensorflow >= 2.10.0
- Pillow >= 9.0.0

---

## ⚠️ Disclaimer

This is an educational project and should **NOT** be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Updated to modern TensorFlow 2.x framework
- Dataset sourced from medical imaging research

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📧 Contact

For questions or suggestions, please open an issue in this repository.
