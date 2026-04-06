from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
from PIL import Image
import pickle
import os
import io
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'csv', 'json', 'wav', 'mp3', 'ogg', 'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for models
behavioral_model = None
behavioral_scaler = None
voice_model = None
facial_model = None
fusion_model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load all trained models"""
    global behavioral_model, behavioral_scaler, voice_model, facial_model, fusion_model
    
    try:
        print("Loading models...")
        
        # Load behavioral model
        if os.path.exists('saved_models/behavioral_model.pkl'):
            with open('saved_models/behavioral_model.pkl', 'rb') as f:
                data = pickle.load(f)
                behavioral_model = data['model']
                behavioral_scaler = data['scaler']
            print("✓ Behavioral model loaded")
        else:
            print("⚠ Behavioral model not found")
        
        # Load voice model (try .h5 then .keras)
        if os.path.exists('saved_models/voice_model.h5'):
            voice_model = keras.models.load_model('saved_models/voice_model.h5')
            print("✓ Voice model loaded (.h5)")
        elif os.path.exists('saved_models/voice_model.keras'):
            voice_model = keras.models.load_model('saved_models/voice_model.keras')
            print("✓ Voice model loaded (.keras)")
        else:
            print("⚠ Voice model not found")
        
        # Load facial model (try .h5 then .keras)
        if os.path.exists('saved_models/facial_model.h5'):
            facial_model = keras.models.load_model('saved_models/facial_model.h5')
            print("✓ Facial model loaded (.h5)")
        elif os.path.exists('saved_models/facial_model.keras'):
            facial_model = keras.models.load_model('saved_models/facial_model.keras')
            print("✓ Facial model loaded (.keras)")
        else:
            print("⚠ Facial model not found")
        
        # Load fusion model (try .h5 then .keras)
        if os.path.exists('saved_models/fusion_model.h5'):
            fusion_model = keras.models.load_model('saved_models/fusion_model.h5')
            print("✓ Fusion model loaded (.h5)")
        elif os.path.exists('saved_models/fusion_model.keras'):
            fusion_model = keras.models.load_model('saved_models/fusion_model.keras')
            print("✓ Fusion model loaded (.keras)")
        else:
            print("⚠ Fusion model not found")
        
        print("Model loading complete!")
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")

def extract_behavioral_features(data):
    """Extract features from behavioral data"""
    try:
        if isinstance(data, dict):
            # Convert dict to array
            features = np.array(list(data.values())).reshape(1, -1)
        else:
            features = np.array(data).reshape(1, -1)
        
        # Scale features
        if behavioral_scaler:
            features = behavioral_scaler.transform(features)
        
        return features
    except Exception as e:
        print(f"Error extracting behavioral features: {str(e)}")
        return None

def extract_voice_features(audio_data, n_mfcc=40, max_length=500):
    """Extract MFCC + delta + delta2 features from audio"""
    try:
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
        
        # Extract MFCCs + deltas (matching train_final.py)
        mfcc_raw = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = mfcc_raw.T
        delta = librosa.feature.delta(mfcc_raw).T
        delta2 = librosa.feature.delta(mfcc_raw, order=2).T
        combined = np.concatenate([mfccs, delta, delta2], axis=1)  # (time, 120)
        
        # Pad or truncate
        if combined.shape[0] < max_length:
            pad_width = max_length - combined.shape[0]
            combined = np.pad(combined, ((0, pad_width), (0, 0)), mode='constant')
        else:
            combined = combined[:max_length, :]
        
        # Normalize
        combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-8)
        
        return np.expand_dims(combined, axis=(0, -1))  # Add batch and channel dims
    except Exception as e:
        print(f"Error extracting voice features: {str(e)}")
        return None

def extract_facial_features(image_data, img_size=(128, 128)):
    """Extract features from facial image"""
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error extracting facial features: {str(e)}")
        return None

def get_embeddings(X_beh, X_voice, X_img):
    """Extract embeddings from all models"""
    try:
        embeddings = []
        
        # Behavioral embeddings
        if behavioral_model is not None:
            beh_emb = behavioral_model.predict_proba(X_beh)
            embeddings.append(beh_emb)
        
        # Voice embeddings
        if voice_model is not None:
            embedding_model_voice = keras.Model(
                inputs=voice_model.input,
                outputs=voice_model.get_layer('embedding').output
            )
            voice_emb = embedding_model_voice.predict(X_voice, verbose=0)
            embeddings.append(voice_emb)
        
        # Facial embeddings
        if facial_model is not None:
            embedding_model_facial = keras.Model(
                inputs=facial_model.input,
                outputs=facial_model.get_layer('embedding').output
            )
            facial_emb = embedding_model_facial.predict(X_img, verbose=0)
            embeddings.append(facial_emb)
        
        if len(embeddings) == 0:
            return None
        
        return np.concatenate(embeddings, axis=1)
    except Exception as e:
        print(f"Error extracting embeddings: {str(e)}")
        return None

# ============= API ROUTES =============

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "Multimodal ASD Screening API",
        "version": "1.0.0"
    }), 200

@app.route('/api/models/status', methods=['GET'])
def model_status():
    """Check which models are loaded"""
    return jsonify({
        "behavioral_model": behavioral_model is not None,
        "voice_model": voice_model is not None,
        "facial_model": facial_model is not None,
        "fusion_model": fusion_model is not None
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict ASD screening from multimodal data
    
    Expected: multipart/form-data with:
    - behavioral_data: CSV or JSON file
    - voice_data: Audio file (WAV, MP3, OGG)
    - image_data: Image file (JPG, PNG)
    """
    try:
        # Check if all files are present
        if 'behavioral_data' not in request.files:
            return jsonify({"error": "Missing behavioral_data"}), 400
        if 'voice_data' not in request.files:
            return jsonify({"error": "Missing voice_data"}), 400
        if 'image_data' not in request.files:
            return jsonify({"error": "Missing image_data"}), 400
        
        behavioral_file = request.files['behavioral_data']
        voice_file = request.files['voice_data']
        image_file = request.files['image_data']
        
        # Validate files
        if behavioral_file.filename == '' or voice_file.filename == '' or image_file.filename == '':
            return jsonify({"error": "Empty filenames"}), 400
        
        if not (allowed_file(behavioral_file.filename) and 
                allowed_file(voice_file.filename) and 
                allowed_file(image_file.filename)):
            return jsonify({"error": "Invalid file types"}), 400
        
        # Extract features
        print("Extracting features...")
        
        # Behavioral features
        if behavioral_file.filename.endswith('.csv'):
            import csv
            import io
            stream = io.StringIO(behavioral_file.read().decode("UTF8"), newline=None)
            behavioral_data = {}
            reader = csv.DictReader(stream)
            for row in reader:
                for key, value in row.items():
                    if key != 'label' and key != 'id':
                        behavioral_data[key] = float(value)
            X_behavioral = extract_behavioral_features(behavioral_data)
        else:
            return jsonify({"error": "Behavioral data must be CSV"}), 400
        
        # Voice features
        voice_data = voice_file.read()
        X_voice = extract_voice_features(voice_data)
        
        # Facial features
        image_data = image_file.read()
        X_facial = extract_facial_features(image_data)
        
        if X_behavioral is None or X_voice is None or X_facial is None:
            return jsonify({"error": "Feature extraction failed"}), 500
        
        # Get individual predictions
        print("Getting predictions from individual models...")
        behavioral_pred = None
        voice_pred = None
        facial_pred = None
        
        if behavioral_model is not None:
            beh_prob = behavioral_model.predict_proba(X_behavioral)[0]
            behavioral_pred = {
                "prediction": "ASD" if beh_prob[1] > 0.5 else "Non-ASD",
                "probability": float(beh_prob[1])
            }
        
        if voice_model is not None:
            voice_prob = voice_model.predict(X_voice, verbose=0)[0]
            voice_pred = {
                "prediction": "ASD" if voice_prob[1] > 0.5 else "Non-ASD",
                "probability": float(voice_prob[1])
            }
        
        if facial_model is not None:
            facial_prob = facial_model.predict(X_facial, verbose=0)[0]
            facial_pred = {
                "prediction": "ASD" if facial_prob[1] > 0.5 else "Non-ASD",
                "probability": float(facial_prob[1])
            }
        
        # Get fusion prediction
        fusion_pred = None
        if fusion_model is not None:
            print("Getting fusion prediction...")
            X_fusion = get_embeddings(X_behavioral, X_voice, X_facial)
            if X_fusion is not None:
                fusion_prob = fusion_model.predict(X_fusion, verbose=0)[0]
                confidence = float(fusion_prob[1])
                
                # Determine risk level
                if confidence < 0.3:
                    risk_level = "Low Risk"
                elif confidence < 0.7:
                    risk_level = "Moderate Risk"
                else:
                    risk_level = "High Risk"
                
                fusion_pred = {
                    "label": "ASD Positive" if confidence > 0.5 else "ASD Negative",
                    "confidence": confidence,
                    "risk_score": confidence
                }
        
        # Prepare response
        response = {
            "success": True,
            "prediction": fusion_pred or {
                "label": "Unable to predict",
                "confidence": 0.5,
                "risk_score": 0.5
            },
            "individual_predictions": {
                "behavioral": behavioral_pred,
                "voice": voice_pred,
                "facial": facial_pred
            },
            "risk_level": risk_level if fusion_pred else "Unknown"
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded"""
    return jsonify({"error": "File size exceeds 50MB limit"}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

# ============= MAIN =============

if __name__ == '__main__':
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Load models
    print("Starting Multimodal ASD Screening API...")
    load_models()
    
    # Run server
    print("\n" + "="*50)
    print("API running at http://localhost:5000")
    print("="*50)
    print("\nEndpoints:")
    print("  GET  /                    - Health check")
    print("  GET  /api/models/status   - Check loaded models")
    print("  POST /api/predict         - Get ASD prediction")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)