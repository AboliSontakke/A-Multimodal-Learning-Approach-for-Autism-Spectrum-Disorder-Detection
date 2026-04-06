import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = {
  /**
   * Predict ASD from multimodal data
   * @param {Object} files - Object containing behavioral, voice, and image files
   * @returns {Promise} - Prediction results
   */
  predictASD: async (files) => {
    try {
      const formData = new FormData();
      formData.append('behavioral_data', files.behavioral);
      formData.append('voice_data', files.voice);
      formData.append('image_data', files.image);

      const response = await axios.post(
        `${API_BASE_URL}/api/predict`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 60000, // 60 second timeout
        }
      );

      if (response.data.success) {
        return response.data;
      } else {
        throw new Error(response.data.error || 'Prediction failed');
      }
    } catch (error) {
      if (error.response) {
        // Server responded with error
        throw new Error(error.response.data.error || 'Server error occurred');
      } else if (error.request) {
        // No response received
        throw new Error('No response from server. Please check your connection.');
      } else {
        // Request setup error
        throw new Error(error.message || 'Failed to make prediction request');
      }
    }
  },

  /**
   * Check model status
   * @returns {Promise} - Model status
   */
  checkModelStatus: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/models/status`);
      return response.data;
    } catch (error) {
      throw new Error('Failed to check model status');
    }
  },

  /**
   * Health check
   * @returns {Promise} - API health status
   */
  healthCheck: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/`);
      return response.data;
    } catch (error) {
      throw new Error('API is not responding');
    }
  },
};

export default api;
