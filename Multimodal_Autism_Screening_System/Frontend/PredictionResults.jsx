import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  LinearProgress,
  Divider,
  Alert
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Refresh
} from '@mui/icons-material';

const PredictionResults = ({ results, onReset }) => {
  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'Low Risk':
        return 'success';
      case 'Moderate Risk':
        return 'warning';
      case 'High Risk':
        return 'error';
      default:
        return 'default';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'Low Risk':
        return <CheckCircle />;
      case 'Moderate Risk':
        return <Warning />;
      case 'High Risk':
        return <ErrorIcon />;
      default:
        return null;
    }
  };

  const formatPercentage = (value) => {
    return (value * 100).toFixed(1) + '%';
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Screening Results
      </Typography>

      {/* Main Prediction */}
      <Card sx={{ mb: 3, backgroundColor: '#f5f5f5' }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Overall Prediction
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4" sx={{ mr: 2 }}>
                  {results.prediction.label}
                </Typography>
                <Chip
                  label={results.risk_level}
                  color={getRiskColor(results.risk_level)}
                  icon={getRiskIcon(results.risk_level)}
                />
              </Box>
              <Typography variant="body2" color="text.secondary">
                Confidence Score: {formatPercentage(results.prediction.confidence)}
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body2" gutterBottom>
                Risk Score
              </Typography>
              <LinearProgress
                variant="determinate"
                value={results.prediction.risk_score * 100}
                sx={{ height: 10, borderRadius: 5 }}
                color={getRiskColor(results.risk_level)}
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                {formatPercentage(results.prediction.risk_score)}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Individual Modality Results */}
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
        Individual Modality Analysis
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                Behavioral Analysis
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box sx={{ my: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Prediction
                </Typography>
                <Typography variant="h6">
                  {results.individual_predictions.behavioral.prediction}
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Probability
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={results.individual_predictions.behavioral.probability * 100}
                  sx={{ height: 8, borderRadius: 5, mb: 1 }}
                />
                <Typography variant="body2">
                  {formatPercentage(results.individual_predictions.behavioral.probability)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                Voice Analysis
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box sx={{ my: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Prediction
                </Typography>
                <Typography variant="h6">
                  {results.individual_predictions.voice.prediction}
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Probability
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={results.individual_predictions.voice.probability * 100}
                  sx={{ height: 8, borderRadius: 5, mb: 1 }}
                />
                <Typography variant="body2">
                  {formatPercentage(results.individual_predictions.voice.probability)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                Facial Analysis
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box sx={{ my: 2 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Prediction
                </Typography>
                <Typography variant="h6">
                  {results.individual_predictions.facial.prediction}
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Probability
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={results.individual_predictions.facial.probability * 100}
                  sx={{ height: 8, borderRadius: 5, mb: 1 }}
                />
                <Typography variant="body2">
                  {formatPercentage(results.individual_predictions.facial.probability)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Disclaimer */}
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Important:</strong> This screening tool is designed to assist healthcare
          professionals and should not be used as a sole diagnostic tool. Please consult
          with a qualified medical professional for proper diagnosis and treatment.
        </Typography>
      </Alert>

      {/* Actions */}
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          startIcon={<Refresh />}
          onClick={onReset}
          size="large"
        >
          New Screening
        </Button>
      </Box>
    </Box>
  );
};

export default PredictionResults;
