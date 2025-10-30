/**
 * API Service for LSTM Text Generation
 * Handles all communication with the FastAPI backend
 */

import axios from 'axios';

// Configure base URL - uses proxy in development, environment variable in production
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('Network Error:', error.message);
    } else {
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

/**
 * Check API health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('Failed to connect to API');
  }
};

/**
 * Get LSTM model status
 * @returns {Promise<Object>} LSTM model status and configuration
 */
export const getLSTMStatus = async () => {
  try {
    const response = await api.get('/lstm/status');
    return response.data;
  } catch (error) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to get LSTM status');
  }
};

/**
 * Generate text using LSTM model
 * @param {string} seedText - Starting text
 * @param {number} numWords - Number of words to generate (1-50)
 * @param {number} temperature - Sampling temperature (0.1-2.0)
 * @returns {Promise<Object>} Generated text result
 */
export const generateText = async (seedText, numWords = 10, temperature = 0.7) => {
  try {
    const response = await api.post('/lstm/generate', {
      seed_text: seedText,
      num_words: numWords,
      temperature: temperature,
    });
    return response.data;
  } catch (error) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to generate text');
  }
};

/**
 * Get example seed phrases for text generation
 * @returns {Promise<Object>} Example seed phrases by category
 */
export const getLSTMExamples = async () => {
  try {
    const response = await api.get('/lstm/examples');
    return response.data;
  } catch (error) {
    throw new Error('Failed to load LSTM examples');
  }
};

export default api;
