# Lung Disease Prediction System

An AI-powered web application for detecting lung diseases from medical images using deep learning.

## Features

- Upload chest X-ray images for disease prediction
- Real-time analysis with confidence scores
- Detailed disease information and recommendations
- PDF report generation
- Interactive dashboard with model metrics and performance visualizations

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV, Pillow
- **PDF Generation**: FPDF
- **Data Visualization**: Matplotlib, Seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lung-disease-prediction.git
   cd lung-disease-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run lung_disease_streamlit.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload a chest X-ray image and get instant predictions

## Project Structure

```
lung-disease-prediction/
├── lung_disease_streamlit.py    # Main Streamlit application
├── d3net_deployment_safe.keras  # Pre-trained Keras model
├── requirements.txt             # Python dependencies
├── class_mapping.json          # Disease class mappings
├── model_summary.csv           # Model performance metrics
├── confusion_matrix_percent.csv # Confusion matrix data
├── test_load.py                # Model loading test script
├── test_model.py               # Model testing script
├── requirements_file.py        # Alternative requirements (legacy)
├── TODO.md                     # Development notes
└── README.md                   # This file
```

## Model Information

- **Architecture**: Custom CNN (D3Net)
- **Input**: Chest X-ray images (various formats supported)
- **Output**: Multi-class classification for lung diseases
- **Classes**: Normal, Bacterial Pneumonia, Tuberculosis, etc.
- **Performance**: High accuracy on validation dataset

## Dataset

The model was trained on a comprehensive dataset of chest X-ray images including:
- Normal cases
- Bacterial pneumonia
- Viral pneumonia
- Tuberculosis
- Other lung conditions

### Data Preparation

Organize your training data in the following directory structure:

```
data/
├── Bacterial_Pneumonia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Tuberculosis/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Training the Model

If you need to retrain the model (recommended for better performance):

1. Prepare your dataset in the directory structure above
2. Run the diagnostic script to check for issues:
   ```bash
   python model_diagnosis.py
   ```
3. Train the model using the fixed training script:
   ```bash
   python train_model_fixed.py
   ```
   When prompted, enter the path to your training data directory.

The training script includes:
- Proper data augmentation to prevent overfitting
- Early stopping and learning rate scheduling
- EfficientNetB0 backbone with ImageNet pretraining
- Regularization techniques
- Comprehensive evaluation and visualization

## Model Issues and Fixes

The original model had reliability issues due to overfitting. The fixes include:

- **Overconfidence Detection**: Model now detects and reports uncertain predictions
- **Improved Training**: New training script with proper regularization
- **Better Architecture**: Uses EfficientNetB0 as intended
- **Data Augmentation**: Prevents overfitting with various transformations
- **Regularization**: L2 regularization and dropout layers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset providers and medical institutions
- TensorFlow/Keras community
- Streamlit for the amazing web framework

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.