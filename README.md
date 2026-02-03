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