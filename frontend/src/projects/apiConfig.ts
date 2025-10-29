/**
 * Centralized API configuration for the application
 * Supports both local development and Render deployment
 */

// Default deployed API URL (Render)
const DEPLOYED_API_URL = "https://cst-435-project-hub.onrender.com";

// Local development API URL
const LOCAL_API_URL = "http://localhost:8000";

/**
 * Get the API base URL
 * Priority:
 * 1. Environment variable REACT_APP_API_URL (set by build system)
 * 2. Try to connect to local API first (for development)
 * 3. Fall back to deployed API (for Render or production)
 */
export const getApiBaseUrl = (): string => {
  // If environment variable is set, use it (takes priority)
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }

  // Default: use deployed API for production/Render
  // Comment out below line to test local API first
  return DEPLOYED_API_URL;

  // Uncomment below for development (tries local first, then deployed)
  // return LOCAL_API_URL;
};

export const API_BASE_URL = getApiBaseUrl();
