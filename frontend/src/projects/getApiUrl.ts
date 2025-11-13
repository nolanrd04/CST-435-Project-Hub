/**
 * Get the appropriate API URL based on user preference
 * Stored in localStorage as 'API_MODE'
 */
export function getApiUrl(): string {
  const apiMode = localStorage.getItem('API_MODE');

  if (apiMode === 'local') {
    return 'http://localhost:8000';
  }

  // Default to deployed API
  return 'https://sorenhaynes.duckdns.org:8443';
}
