// ===========================================================================
// Multimodal ASD Screening System — Frontend Script
// ===========================================================================
// Connects to Flask backend at /api/predict for real ML predictions
// ===========================================================================

const API_BASE = 'http://localhost:5000';

let uploadedFiles = {
    behavioral: null,
    voice: null,
    image: null
};

// ===========================================================================
// Navigation & UI
// ===========================================================================

function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

// ===========================================================================
// File Upload Handling
// ===========================================================================

function handleFileSelect(type, input) {
    const file = input.files[0];
    if (!file) return;

    const statusElement = document.getElementById(`${type}-status`);

    const validation = validateFile(file, type);
    if (!validation.valid) {
        statusElement.className = 'file-status error';
        statusElement.innerHTML = `<i class="fas fa-times-circle"></i> ${validation.message}`;
        input.value = '';
        return;
    }

    uploadedFiles[type] = file;

    statusElement.className = 'file-status success';
    statusElement.innerHTML = `<i class="fas fa-check-circle"></i> ${file.name} (${formatFileSize(file.size)})`;

    if (type === 'image') {
        const previewElement = document.getElementById('image-preview');
        const reader = new FileReader();
        reader.onload = function(e) {
            previewElement.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }

    checkAllFilesUploaded();
}

function validateFile(file, type) {
    const maxSizes = {
        behavioral: 10 * 1024 * 1024,
        voice: 30 * 1024 * 1024,
        image: 20 * 1024 * 1024
    };

    const allowedTypes = {
        behavioral: ['.csv', '.json'],
        voice: ['.wav', '.mp3', '.ogg'],
        image: ['.jpg', '.jpeg', '.png']
    };

    if (file.size > maxSizes[type]) {
        return { valid: false, message: `File size exceeds ${formatFileSize(maxSizes[type])} limit` };
    }

    const fileName = file.name.toLowerCase();
    const validType = allowedTypes[type].some(ext => fileName.endsWith(ext));

    if (!validType) {
        return { valid: false, message: `Invalid file type. Accepted: ${allowedTypes[type].join(', ')}` };
    }

    return { valid: true };
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function checkAllFilesUploaded() {
    const allUploaded = uploadedFiles.behavioral && uploadedFiles.voice && uploadedFiles.image;
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.disabled = !allUploaded;
}

function clearAll() {
    uploadedFiles = { behavioral: null, voice: null, image: null };

    document.getElementById('behavioral-file').value = '';
    document.getElementById('voice-file').value = '';
    document.getElementById('image-file').value = '';

    document.getElementById('behavioral-status').innerHTML = '';
    document.getElementById('voice-status').innerHTML = '';
    document.getElementById('image-status').innerHTML = '';
    document.getElementById('image-preview').innerHTML = '';

    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('results-section').style.display = 'none';
}

// ===========================================================================
// API Integration — Real Backend Calls
// ===========================================================================

async function analyzeData() {
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.disabled = true;

    document.getElementById('upload-form').style.display = 'none';

    const loadingScreen = document.getElementById('loading-screen');
    loadingScreen.style.display = 'block';

    // Show progress steps
    const steps = [
        { text: 'Uploading files to server...', progress: 15 },
        { text: 'Processing behavioral data...', progress: 30 },
        { text: 'Extracting audio features (MFCCs)...', progress: 50 },
        { text: 'Analyzing facial image...', progress: 70 },
        { text: 'Running fusion network...', progress: 85 },
    ];

    let stepIndex = 0;
    const progressInterval = setInterval(() => {
        if (stepIndex < steps.length) {
            document.getElementById('loading-text').textContent = steps[stepIndex].text;
            document.getElementById('progress').style.width = steps[stepIndex].progress + '%';
            stepIndex++;
        }
    }, 1200);

    try {
        // Build FormData with the uploaded files
        const formData = new FormData();
        formData.append('behavioral_data', uploadedFiles.behavioral);
        formData.append('voice_data', uploadedFiles.voice);
        formData.append('image_data', uploadedFiles.image);

        // Call the backend API
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error (${response.status})`);
        }

        const data = await response.json();

        document.getElementById('loading-text').textContent = 'Generating results...';
        document.getElementById('progress').style.width = '100%';

        // Small delay to show 100% progress
        await new Promise(resolve => setTimeout(resolve, 500));

        if (data.success) {
            showResults(data);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }

    } catch (error) {
        clearInterval(progressInterval);
        console.error('API Error:', error);

        // Show error and fall back to simulated results
        document.getElementById('loading-text').textContent =
            `⚠ Backend unavailable (${error.message}). Showing simulated results...`;
        document.getElementById('progress').style.width = '100%';

        await new Promise(resolve => setTimeout(resolve, 2000));

        // Fall back to simulated results so the UI still works
        const simulatedData = generateSimulatedResults();
        showResults(simulatedData);
    }
}

// ===========================================================================
// Show Results (from real API or simulated fallback)
// ===========================================================================

function showResults(data) {
    document.getElementById('loading-screen').style.display = 'none';

    // Parse results from backend API response format
    let results;

    if (data.prediction && data.individual_predictions) {
        // Real API response
        const confidence = data.prediction.confidence || 0.5;
        let riskLevel, riskClass;

        if (confidence < 0.3) {
            riskLevel = 'Low Risk'; riskClass = 'low';
        } else if (confidence < 0.7) {
            riskLevel = 'Moderate Risk'; riskClass = 'moderate';
        } else {
            riskLevel = 'High Risk'; riskClass = 'high';
        }

        results = {
            overall: data.prediction.label || (confidence > 0.5 ? 'ASD Positive' : 'ASD Negative'),
            riskLevel: data.risk_level || riskLevel,
            riskClass: riskClass,
            confidence: confidence,
            behavioral: data.individual_predictions.behavioral?.probability || 0.5,
            voice: data.individual_predictions.voice?.probability || 0.5,
            facial: data.individual_predictions.facial?.probability || 0.5,
            source: 'backend'
        };
    } else {
        // Simulated fallback
        results = data;
    }

    updateResultsUI(results);

    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function updateResultsUI(results) {
    document.getElementById('prediction-label').textContent = results.overall;

    const riskBadge = document.getElementById('risk-badge');
    riskBadge.textContent = results.riskLevel;
    riskBadge.className = `risk-badge ${results.riskClass}`;

    const confidencePercent = Math.round(results.confidence * 100);
    document.getElementById('confidence-bar').style.width = confidencePercent + '%';
    document.getElementById('confidence-text').textContent = confidencePercent + '%';

    updateModalityResult('behavioral', results.behavioral);
    updateModalityResult('voice', results.voice);
    updateModalityResult('facial', results.facial);

    // Show data source badge
    const sourceInfo = document.getElementById('source-info');
    if (sourceInfo) {
        if (results.source === 'backend') {
            sourceInfo.innerHTML = '<i class="fas fa-check-circle" style="color: #4CAF50;"></i> Results from trained ML models';
            sourceInfo.style.display = 'block';
        } else {
            sourceInfo.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: #FF9800;"></i> Simulated results (backend offline)';
            sourceInfo.style.display = 'block';
        }
    }
}

function updateModalityResult(type, probability) {
    const percent = Math.round(probability * 100);
    const prediction = probability > 0.5 ? 'ASD' : 'Non-ASD';

    document.getElementById(`${type}-prediction`).textContent = prediction;
    document.getElementById(`${type}-bar`).style.width = percent + '%';
    document.getElementById(`${type}-prob`).textContent = percent + '%';
}

// ===========================================================================
// Simulated Results (fallback when backend is offline)
// ===========================================================================

function generateSimulatedResults() {
    const risk = Math.random();
    let riskLevel, riskClass, overallPrediction;

    if (risk < 0.3) {
        riskLevel = 'Low Risk'; riskClass = 'low'; overallPrediction = 'ASD Negative';
    } else if (risk < 0.7) {
        riskLevel = 'Moderate Risk'; riskClass = 'moderate'; overallPrediction = 'ASD Positive';
    } else {
        riskLevel = 'High Risk'; riskClass = 'high'; overallPrediction = 'ASD Positive';
    }

    const variance = 0.1;
    const behavioral = Math.min(Math.max(risk + (Math.random() - 0.5) * variance, 0.1), 0.95);
    const voice = Math.min(Math.max(risk + (Math.random() - 0.5) * variance, 0.1), 0.95);
    const facial = Math.min(Math.max(risk + (Math.random() - 0.5) * variance, 0.1), 0.95);

    return {
        overall: overallPrediction,
        riskLevel: riskLevel,
        riskClass: riskClass,
        confidence: (behavioral + voice + facial) / 3,
        behavioral: behavioral,
        voice: voice,
        facial: facial,
        source: 'simulated'
    };
}

// ===========================================================================
// Download & New Analysis
// ===========================================================================

function downloadResults() {
    const results = {
        timestamp: new Date().toISOString(),
        prediction: document.getElementById('prediction-label').textContent,
        riskLevel: document.getElementById('risk-badge').textContent,
        confidence: document.getElementById('confidence-text').textContent,
        behavioral: {
            prediction: document.getElementById('behavioral-prediction').textContent,
            probability: document.getElementById('behavioral-prob').textContent
        },
        voice: {
            prediction: document.getElementById('voice-prediction').textContent,
            probability: document.getElementById('voice-prob').textContent
        },
        facial: {
            prediction: document.getElementById('facial-prediction').textContent,
            probability: document.getElementById('facial-prob').textContent
        }
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(results, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "asd_screening_results.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

function newAnalysis() {
    clearAll();
    document.getElementById('upload-form').style.display = 'block';
    scrollToSection('upload');
}

// ===========================================================================
// Check Backend Status on Page Load
// ===========================================================================

async function checkBackendStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/models/status`, { 
            signal: AbortSignal.timeout(3000)
        });
        if (response.ok) {
            const data = await response.json();
            const modelsLoaded = Object.values(data).filter(v => v === true).length;
            console.log(`✓ Backend connected — ${modelsLoaded}/4 models loaded`);

            const statusEl = document.getElementById('backend-status');
            if (statusEl) {
                statusEl.innerHTML = `<i class="fas fa-check-circle" style="color:#4CAF50"></i> Backend connected (${modelsLoaded}/4 models)`;
                statusEl.style.display = 'inline-block';
            }
        }
    } catch (err) {
        console.log('⚠ Backend offline — will use simulated results as fallback');
        const statusEl = document.getElementById('backend-status');
        if (statusEl) {
            statusEl.innerHTML = '<i class="fas fa-exclamation-triangle" style="color:#FF9800"></i> Backend offline (simulated mode)';
            statusEl.style.display = 'inline-block';
        }
    }
}

// ===========================================================================
// Page Initialization
// ===========================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Check backend connectivity
    checkBackendStatus();

    // Active section highlighting on scroll
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (window.pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            scrollToSection(targetId);
        });
    });
});