import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Button,
  Alert
} from '@mui/material';
import UploadForm from './components/UploadForm';
import PredictionResults from './components/PredictionResults';
import './App.css';

const steps = ['Upload Data', 'Analysis', 'Results'];

function App() {
  const [activeStep, setActiveStep] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState({
    behavioral: null,
    voice: null,
    image: null
  });
  const [predictionResults, setPredictionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setUploadedFiles({ behavioral: null, voice: null, image: null });
    setPredictionResults(null);
    setError(null);
  };

  const handleFilesUploaded = (files) => {
    setUploadedFiles(files);
    handleNext();
  };

  const handlePredictionComplete = (results) => {
    setPredictionResults(results);
    setLoading(false);
    handleNext();
  };

  const handleError = (errorMsg) => {
    setError(errorMsg);
    setLoading(false);
  };

  return (
    <div className="App">
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          {/* Header */}
          <Paper elevation={3} sx={{ p: 3, mb: 4, backgroundColor: '#1976d2', color: 'white' }}>
            <Typography variant="h3" component="h1" gutterBottom>
              Multimodal ASD Screening System
            </Typography>
            <Typography variant="h6">
              AI-Powered Autism Spectrum Disorder Screening
            </Typography>
          </Paper>

          {/* Stepper */}
          <Box sx={{ mb: 4 }}>
            <Stepper activeStep={activeStep}>
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </Box>

          {/* Error Alert */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {/* Content */}
          <Paper elevation={3} sx={{ p: 3 }}>
            {activeStep === 0 && (
              <UploadForm
                onFilesUploaded={handleFilesUploaded}
                uploadedFiles={uploadedFiles}
                setUploadedFiles={setUploadedFiles}
              />
            )}

            {activeStep === 1 && (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography variant="h5" gutterBottom>
                  Analyzing Multimodal Data...
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph>
                  Processing behavioral questionnaire, voice recording, and facial image.
                </Typography>
                <Box sx={{ mt: 3 }}>
                  <Button
                    variant="contained"
                    onClick={() => {
                      setLoading(true);
                      // Import and use API service
                      import('./services/api').then((api) => {
                        api.default.predictASD(uploadedFiles)
                          .then(handlePredictionComplete)
                          .catch((err) => handleError(err.message));
                      });
                    }}
                    disabled={loading}
                    size="large"
                  >
                    {loading ? 'Processing...' : 'Start Analysis'}
                  </Button>
                </Box>
                <Box sx={{ mt: 2 }}>
                  <Button onClick={handleBack}>
                    Back
                  </Button>
                </Box>
              </Box>
            )}

            {activeStep === 2 && predictionResults && (
              <PredictionResults results={predictionResults} onReset={handleReset} />
            )}
          </Paper>

          {/* Footer */}
          <Box sx={{ mt: 4, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              © 2025 Multimodal ASD Screening System | AI in Healthcare Project
            </Typography>
          </Box>
        </Box>
      </Container>
    </div>
  );
}

export default App;
