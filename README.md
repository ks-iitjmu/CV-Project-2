# 🤟 Real-Time Sign Language Recognition System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An advanced computer vision system for real-time American Sign Language (ASL) recognition using hand landmark detection and machine learning.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Project Structure](#-project-structure)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project implements a **real-time sign language recognition system** that can identify **36 different hand signs**:
- 🔤 **26 letters** (A-Z) of the American Sign Language alphabet
- 🔢 **10 digits** (0-9)

The system uses **MediaPipe** for hand landmark detection and a **Random Forest classifier** trained on hand gesture features to recognize signs with high accuracy.

---

## ✨ Features

- 🎥 **Real-time Recognition**: Instant sign language detection through webcam
- 🤖 **Machine Learning**: Random Forest classifier with 200 estimators for robust predictions
- 👋 **Hand Tracking**: 21-point hand landmark detection using Google's MediaPipe
- 📊 **Large Dataset**: Supports 200+ images per class for improved accuracy
- 🎯 **High Accuracy**: Optimized hyperparameters for maximum performance
- 🚀 **Easy to Use**: Simple pipeline from data collection to inference
- 📦 **Modular Design**: Clean separation of data collection, training, and inference

---

## 🎬 Demo

### Recognition in Action
The system displays:
- ✅ Real-time hand landmark tracking
- 🔲 Bounding box around detected hand
- 🔤 Predicted letter/digit above the hand

### Supported Signs
- **Letters**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Linux/macOS/Windows

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd project2
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy==1.26.4` - Numerical computing
- `mediapipe==0.10.11` - Hand landmark detection
- `opencv-python==4.9.0.80` - Computer vision operations
- `scikit-learn==1.4.2` - Machine learning algorithms
- `pillow==10.2.0` - Image processing for GUI
- `pyttsx3==2.90` - Text-to-speech conversion

---

## 📖 Usage

The system consists of **4 main scripts** that form a complete pipeline:

### 1️⃣ Collect Training Images

Collect 200 images for each of the 36 sign classes:

```bash
python3 p_collect_images.py
```

**Instructions**:
- Position your hand to form the sign (A, B, C, etc.)
- Press **'Q'** when ready
- Hold the sign steady while 200 images are captured
- Repeat for all 36 signs

**Tips**:
- 💡 Ensure good lighting
- 🖐️ Keep hand centered in frame
- 🔄 Vary hand position slightly for diversity
- ⏱️ Each class takes ~10 seconds to capture

---

### 2️⃣ Create Feature Dataset

Extract hand landmarks from collected images:

```bash
python3 p_create_dataset.py
```

**What it does**:
- Processes all images in `./data/` directory
- Detects hand landmarks using MediaPipe
- Normalizes coordinates relative to hand position
- Saves features to `data.pickle`

**Output**: Creates `data.pickle` containing normalized hand landmark features

---

### 3️⃣ Train the Classifier

Train the Random Forest model on extracted features:

```bash
python3 p_train_classifier.py
```

**Training Configuration**:
- Algorithm: Random Forest Classifier
- Number of trees: 200
- Max depth: 20
- Train/Test split: 80/20
- Cross-validation: Stratified sampling

**Output**: 
- Displays accuracy score (e.g., "95.5% of samples were classified correctly!")
- Saves trained model to `model.p`

---

### 4️⃣ Run Real-Time Recognition

Start the real-time sign language recognition:

```bash
python3 p_inference_classifier.py
```

**Features**:
- 🎥 Live webcam feed
- 👋 Real-time hand landmark visualization
- 🔲 Bounding box around detected hand
- 🔤 Predicted sign displayed above hand
- ⌨️ Press any key to exit

---

### 5️⃣ Run Sign Language to Speech Conversion (NEW!)

Start the complete sign-to-speech application with GUI:

```bash
python3 p_sign_to_speech.py
```

**Features**:
- 🎥 Live webcam feed with hand tracking
- 🖐️ Real-time hand landmark visualization
- 🔤 Character-by-character text building
- 📝 Sentence construction from recognized signs
- 🔊 Text-to-speech conversion
- 🎨 Clean, user-friendly GUI interface
- ⌨️ Manual controls: Space, Backspace, Clear
- 🗣️ Speak button for audio output

**Interface Components**:
- **Video Feed**: Shows live camera with hand detection
- **Hand Landmarks**: Visual representation of detected hand landmarks
- **Character Display**: Current recognized character
- **Sentence Builder**: Accumulated text from recognized signs
- **Control Buttons**:
  - `Clear`: Reset sentence
  - `Speak`: Convert text to speech
  - `Space`: Add space between words
  - `Backspace`: Remove last character

---

## 🔬 How It Works

### Architecture Overview

```
📷 Webcam Input
    ↓
👋 MediaPipe Hand Detection (21 Landmarks)
    ↓
📐 Feature Extraction (42 Normalized Coordinates)
    ↓
🤖 Random Forest Classifier (200 Trees)
    ↓
🔤 Predicted Sign Output
```

### Technical Details

#### 1. Hand Landmark Detection
- Uses **MediaPipe Hands** solution
- Detects **21 key points** on each hand:
  - Wrist
  - Thumb (4 points)
  - Index finger (4 points)
  - Middle finger (4 points)
  - Ring finger (4 points)
  - Pinky finger (4 points)

#### 2. Feature Engineering
- Extracts X and Y coordinates for all 21 landmarks
- **Normalization**: Subtracts minimum X and Y values
- Creates **42 features per sample** (21 points × 2 coordinates)
- Makes the model **translation-invariant**

#### 3. Classification Model
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - `n_estimators=200`: 200 decision trees
  - `max_depth=20`: Maximum tree depth
  - `random_state=42`: For reproducibility
  - `n_jobs=-1`: Parallel processing

#### 4. Class Mapping
```python
Classes 0-25  → A-Z (chr(65+i))
Classes 26-35 → 0-9 (str(i-26))
```

---

## 📁 Project Structure

```
project2/
│
├── 📄 p_collect_images.py        # Step 1: Collect training images
├── 📄 p_create_dataset.py        # Step 2: Extract hand landmarks
├── 📄 p_train_classifier.py      # Step 3: Train ML model
├── 📄 p_inference_classifier.py  # Step 4: Real-time recognition
├── 📄 p_sign_to_speech.py        # Step 5: Sign-to-Speech GUI Application (NEW!)
│
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Documentation (this file)
│
├── 📦 model.p                     # Trained classifier (generated)
├── 📦 data.pickle                 # Feature dataset (generated)
│
└── 📁 data/                       # Training images directory
    ├── 0/   # Class 0 (Letter A)
    ├── 1/   # Class 1 (Letter B)
    ├── 2/   # Class 2 (Letter C)
    ├── ...
    ├── 25/  # Class 25 (Letter Z)
    ├── 26/  # Class 26 (Digit 0)
    ├── ...
    └── 35/  # Class 35 (Digit 9)
```

---

## 📊 Model Performance

### Expected Accuracy
- **Training Accuracy**: ~98-99%
- **Test Accuracy**: ~90-95%
- **Real-time Performance**: 30+ FPS

### Performance Factors
| Factor | Impact |
|--------|--------|
| 💡 **Lighting** | High - Affects hand detection |
| 📷 **Camera Quality** | Medium - Better resolution helps |
| 🖐️ **Hand Position** | High - Centered hands work best |
| 🎯 **Sign Precision** | High - Clear signs improve accuracy |
| 🔄 **Training Data Variety** | Critical - More diverse = better |

### Optimization Tips
1. **Increase dataset_size** in `p_collect_images.py` (e.g., 300-500 images)
2. **Vary hand positions** during data collection
3. **Use consistent lighting** during training and inference
4. **Tune model hyperparameters** in `p_train_classifier.py`
5. **Adjust detection confidence** in MediaPipe settings

---

## 🔧 Troubleshooting

### Common Issues

#### ❌ "ValueError: setting an array element with a sequence"
**Cause**: Inconsistent feature dimensions (multiple hands detected)

**Solution**: Ensure only one hand is visible during data collection

#### ❌ Camera not working
**Cause**: Permission issues or wrong camera index

**Solution**: 
```python
# In any script with cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)  # Try different indices: 0, 1, 2
```

#### ❌ Low accuracy
**Cause**: Insufficient or poor-quality training data

**Solutions**:
- Collect more images per class (increase `dataset_size`)
- Ensure consistent hand positioning
- Improve lighting conditions
- Retrain with better data

#### ❌ "ModuleNotFoundError"
**Cause**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

#### ❌ Slow performance
**Cause**: CPU bottleneck

**Solutions**:
- Reduce video resolution
- Decrease MediaPipe detection confidence
- Use fewer trees in Random Forest (e.g., `n_estimators=100`)

---

## 🚀 Future Enhancements

### Planned Features
- [ ] 🎥 Support for dynamic signs (motion-based)
- [ ] 🤝 Two-handed sign recognition
- [ ] 🌐 Web interface for easy access
- [ ] 📱 Mobile app deployment
- [ ] 🧠 Deep learning model (CNN/LSTM)
- [ ] 🗣️ Text-to-speech for recognized signs
- [ ] 📚 Expanded sign language support (BSL, ISL, etc.)
- [ ] 📊 Real-time accuracy metrics display
- [ ] 💾 Cloud-based model training
- [ ] 🎮 Interactive learning mode

### Advanced Improvements
- **Data Augmentation**: Rotation, scaling, brightness variations
- **Deep Learning**: CNN-based feature extraction
- **Temporal Models**: LSTM for sign sequences
- **Transfer Learning**: Pre-trained hand pose models
- **Edge Deployment**: TensorFlow Lite for mobile/embedded systems

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌱 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

### Areas for Contribution
- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New features
- 🎨 UI/UX enhancements
- 🧪 Test coverage
- 🌍 Internationalization

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **Google MediaPipe** - For excellent hand tracking technology
- **OpenCV** - For computer vision tools
- **scikit-learn** - For machine learning algorithms
- **ASL Community** - For sign language resources and inspiration

---

## 📚 References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [American Sign Language Alphabet](https://www.nidcd.nih.gov/health/american-sign-language)
- [Random Forest Algorithm](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ and Python**

[⬆ Back to Top](#-real-time-sign-language-recognition-system)

</div>
