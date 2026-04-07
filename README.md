# Multimodal ASD Screening System

##  Project Overview

An AI-powered system for Autism Spectrum Disorder (ASD) screening using multimodal deep learning. The system analyzes three types of data:
- **Behavioral questionnaire responses** (structured data)
- **Voice recordings** (audio data with MFCCs)
- **Facial images** (visual data)

By combining all three modalities through a fusion network, the system provides accurate, non-invasive ASD screening with improved performance over single-modality approaches.

##  Architecture

### Backend (Flask + Python)
- **Behavioral Model**: XGBoost classifier for questionnaire data
- **Voice Model**: CNN-LSTM network for audio feature extraction and classification
- **Facial Model**: ResNet50/EfficientNet for facial image analysis
- **Fusion Model**: Dense neural network combining embeddings from all modalities

### Frontend (React)
- Material-UI components for modern, responsive interface
- Multi-step upload process for three data modalities
- Real-time prediction results with risk assessment
- Individual modality analysis display

##  Dataset

**Source**: Multimodal Autism Dataset (Kaggle)

**Modalities**:
1. **Behavioral Data**: Questionnaire responses (CSV/JSON format)
2. **Voice Data**: Audio recordings (.wav, .mp3, .ogg)
3. **Facial Data**: Facial images (.jpg, .png)

**Labels**: Binary classification (ASD Positive / ASD Negative)

##  Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- GPU recommended for training (optional)

### Backend Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multimodal-asd-screening
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p saved_models uploads data
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file `.env`:
```
REACT_APP_API_URL=http://localhost:5000
```

##  Training Models

### Step 1: Download Dataset

Download the Multimodal Autism Dataset from Kaggle:
```bash
kaggle datasets download -d <dataset-name>
unzip <dataset-name>.zip -d data/
```

### Step 2: Run Training Pipeline

Execute the complete training script:
```bash
python complete_training_pipeline.py
```

This will:
- Preprocess all three modalities
- Train behavioral model (XGBoost)
- Train voice model (CNN-LSTM)
- Train facial model (ResNet50)
- Train fusion model
- Save all models to `saved_models/`
- Generate evaluation metrics

### Expected Training Time
- **Behavioral Model**: 5-10 minutes
- **Voice Model**: 1-2 hours (with GPU)
- **Facial Model**: 2-3 hours (with GPU)
- **Fusion Model**: 30 minutes

##  Running the Application

### Start Backend Server

```bash
python app.py
```

Server will start at `http://localhost:5000`

### Start Frontend Development Server

```bash
cd frontend
npm start
```

Application will open at `http://localhost:3000`

##  API Documentation

### POST /api/predict

Perform ASD screening prediction.

**Request**:
- `behavioral_data`: CSV or JSON file with questionnaire responses
- `voice_data`: Audio file (.wav, .mp3, .ogg)
- `image_data`: Facial image (.jpg, .png)

**Response**:
```json
{
  "success": true,
  "prediction": {
    "label": "ASD Positive",
    "confidence": 0.87,
    "risk_score": 0.87
  },
  "individual_predictions": {
    "behavioral": {
      "probability": 0.82,
      "prediction": "ASD"
    },
    "voice": {
      "probability": 0.79,
      "prediction": "ASD"
    },
    "facial": {
      "probability": 0.85,
      "prediction": "ASD"
    }
  },
  "risk_level": "High Risk"
}
```

### GET /api/models/status

Check if models are loaded.

**Response**:
```json
{
  "behavioral_model": true,
  "voice_model": true,
  "facial_model": true,
  "fusion_model": true
}
```

##  Model Performance

### Target Metrics
- **Accuracy**: ≥ 90%
- **Precision**: ≥ 88%
- **Recall**: ≥ 85%
- **F1-Score**: ≥ 87%
- **AUROC**: ≥ 0.85

### Evaluation Metrics
All models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Area Under ROC Curve (AUROC)

##  Project Structure
```
multimodal-asd-screening/
├── backend/
│   ├── app.py                          # Main Flask application
│   ├── config.py                       # Configuration
│   ├── requirements.txt                # Python dependencies
│   ├── models/
│   │   ├── behavioral_model.py         # XGBoost model
│   │   ├── voice_model.py              # CNN-LSTM model
│   │   ├── facial_model.py             # ResNet50 model
│   │   └── fusion_model.py             # Fusion network
│   ├── utils/
│   │   ├── data_preprocessing.py
│   │   ├── audio_features.py
│   │   └── image_processing.py
│   └── saved_models/                   # Trained model weights
│
├── frontend/
│   ├── package.json                    # NPM dependencies
│   ├── public/
│   └── src/
│       ├── App.js                      # Main React component
│       ├── components/
│       │   ├── UploadForm.jsx          # File upload interface
│       │   ├── PredictionResults.jsx   # Results display
│       │   ├── Dashboard.jsx
│       │   └── Header.jsx
│       └── services/
│           └── api.js                  # API service
│
├── notebooks/
│   └── complete_training_pipeline.py   # Complete training script
│
├── data/                               # Dataset directory
├── uploads/                            # Temporary upload storage
└── README.md
```

##  Technologies Used

### Backend
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning models
- **PyTorch**: Alternative DL framework
- **Scikit-learn**: ML algorithms
- **XGBoost**: Gradient boosting
- **librosa**: Audio processing
- **OpenCV**: Image processing

### Frontend
- **React**: UI framework
- **Material-UI**: Component library
- **Axios**: HTTP client
- **Recharts**: Data visualization

##  Key Features

1. **Multimodal Analysis**: Combines three different data types for robust predictions
2. **Individual Insights**: Shows predictions from each modality separately
3. **Risk Assessment**: Provides Low/Moderate/High risk categorization
4. **User-Friendly Interface**: Simple 3-step process for data upload and analysis
5. **Scalable Architecture**: Designed for easy deployment and scaling
6. **Explainable AI**: Individual modality contributions visible to users

##  Model Details

### Behavioral Model (XGBoost)
- **Input**: Questionnaire responses (structured features)
- **Architecture**: Gradient Boosting Trees
- **Output**: ASD probability + 2D embedding

### Voice Model (CNN-LSTM)
- **Input**: MFCC features (40 coefficients, 500 time steps)
- **Architecture**: 
  - 3 CNN layers for spatial features
  - 2 Bi-LSTM layers for temporal patterns
  - Dense layers for classification
- **Output**: ASD probability + 32D embedding

### Facial Model (ResNet50)
- **Input**: Facial images (224×224×3)
- **Architecture**: 
  - Pretrained ResNet50 base
  - Fine-tuned dense layers
  - Global average pooling
- **Output**: ASD probability + 64D embedding

### Fusion Model
- **Input**: Concatenated embeddings (2 + 32 + 64 = 98D)
- **Architecture**: Dense neural network with dropout
- **Output**: Final ASD prediction

##  Future Enhancements

- [ ] Add more modalities (eye-tracking, EEG data)
- [ ] Implement attention visualization
- [ ] Add explainability features (SHAP, Grad-CAM)
- [ ] Support for multiple languages
- [ ] Mobile application version
- [ ] Real-time video analysis
- [ ] Integration with electronic health records
- [ ] Longitudinal tracking of patients


##  References

1. Autism Spectrum Disorder (ASD) - CDC Guidelines
2. Multimodal Deep Learning for Healthcare Applications
3. ResNet: Deep Residual Learning for Image Recognition
4. LSTM Networks for Sequence Modeling
5. XGBoost: A Scalable Tree Boosting System

## 🤝 Acknowledgments

- Kaggle for providing the Multimodal Autism Dataset
- TensorFlow and PyTorch teams
- Open-source community

