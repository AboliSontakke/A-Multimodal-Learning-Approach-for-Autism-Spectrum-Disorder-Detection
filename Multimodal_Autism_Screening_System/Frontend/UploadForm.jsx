import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Alert
} from '@mui/material';
import {
  CloudUpload,
  Description,
  AudioFile,
  Image as ImageIcon,
  CheckCircle
} from '@mui/icons-material';

const UploadForm = ({ onFilesUploaded, uploadedFiles, setUploadedFiles }) => {
  const [errors, setErrors] = useState({});

  const handleFileChange = (fileType) => (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file
      const validationError = validateFile(file, fileType);
      if (validationError) {
        setErrors({ ...errors, [fileType]: validationError });
        return;
      }

      // Clear error and set file
      setErrors({ ...errors, [fileType]: null });
      setUploadedFiles({ ...uploadedFiles, [fileType]: file });
    }
  };

  const validateFile = (file, fileType) => {
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (file.size > maxSize) {
      return 'File size exceeds 50MB limit';
    }

    const validExtensions = {
      behavioral: ['csv', 'json'],
      voice: ['wav', 'mp3', 'ogg'],
      image: ['jpg', 'jpeg', 'png']
    };

    const extension = file.name.split('.').pop().toLowerCase();
    if (!validExtensions[fileType].includes(extension)) {
      return `Invalid file type. Accepted: ${validExtensions[fileType].join(', ')}`;
    }

    return null;
  };

  const handleSubmit = () => {
    // Validate all files are uploaded
    if (!uploadedFiles.behavioral || !uploadedFiles.voice || !uploadedFiles.image) {
      setErrors({ general: 'Please upload all three required files' });
      return;
    }

    onFilesUploaded(uploadedFiles);
  };

  const FileUploadCard = ({ title, description, icon, fileType, acceptedFormats }) => (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {icon}
          <Typography variant="h6" sx={{ ml: 1 }}>
            {title}
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" paragraph>
          {description}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Accepted formats: {acceptedFormats}
        </Typography>
        {uploadedFiles[fileType] && (
          <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', color: 'success.main' }}>
            <CheckCircle sx={{ mr: 1, fontSize: 20 }} />
            <Typography variant="body2">
              {uploadedFiles[fileType].name}
            </Typography>
          </Box>
        )}
        {errors[fileType] && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {errors[fileType]}
          </Alert>
        )}
      </CardContent>
      <CardActions>
        <Button
          variant="contained"
          component="label"
          startIcon={<CloudUpload />}
          fullWidth
        >
          {uploadedFiles[fileType] ? 'Change File' : 'Upload File'}
          <input
            type="file"
            hidden
            accept={acceptedFormats.split(', ').map(f => `.${f}`).join(',')}
            onChange={handleFileChange(fileType)}
          />
        </Button>
      </CardActions>
    </Card>
  );

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Upload Multimodal Data
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Please upload all three types of data for accurate ASD screening.
      </Typography>

      {errors.general && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {errors.general}
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <FileUploadCard
            title="Behavioral Data"
            description="Upload questionnaire responses in CSV or JSON format"
            icon={<Description color="primary" />}
            fileType="behavioral"
            acceptedFormats="csv, json"
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <FileUploadCard
            title="Voice Recording"
            description="Upload audio file of voice sample"
            icon={<AudioFile color="primary" />}
            fileType="voice"
            acceptedFormats="wav, mp3, ogg"
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <FileUploadCard
            title="Facial Image"
            description="Upload clear facial photograph"
            icon={<ImageIcon color="primary" />}
            fileType="image"
            acceptedFormats="jpg, jpeg, png"
          />
        </Grid>
      </Grid>

      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleSubmit}
          disabled={!uploadedFiles.behavioral || !uploadedFiles.voice || !uploadedFiles.image}
        >
          Continue to Analysis
        </Button>
      </Box>
    </Box>
  );
};

export default UploadForm;
