import Constants from 'expo-constants';
import { Platform } from 'react-native';

const BACKEND_PORT = 8000;

// ============================================================
// MANUAL OVERRIDE - Set this if auto-detection doesn't work
// ============================================================
// Uncomment and set your computer's IP address if needed:
// const MANUAL_BACKEND_URL = 'http://192.168.1.100:8000';
// Or for ngrok tunnel:
// const MANUAL_BACKEND_URL = 'https://xxxx-xx-xx-xx-xx.ngrok-free.app';
const MANUAL_BACKEND_URL: string | null = null;
// ============================================================

/**
 * Dynamically get the backend API URL.
 *
 * Strategy: Use the same IP that Expo's Metro bundler uses.
 * If the app loaded successfully, that IP is reachable from this device.
 */
const getApiUrl = (): string => {
  // Check for manual override first
  if (MANUAL_BACKEND_URL) {
    console.log('[API Config] Using manual override:', MANUAL_BACKEND_URL);
    return MANUAL_BACKEND_URL;
  }

  // Production - Railway deployment
  if (!__DEV__) {
    return 'https://dermaintel-production.up.railway.app';
  }

  // Web platform - use same host as the page
  if (Platform.OS === 'web' && typeof window !== 'undefined') {
    const host = window.location.hostname;
    const url = `http://${host}:${BACKEND_PORT}`;
    console.log('[API Config] Web platform, using:', url);
    return url;
  }

  // Get the host from Expo - this is the IP the app used to download its bundle
  const hostUri =
    Constants.expoConfig?.hostUri ||  // Expo SDK 49+
    (Constants as any).manifest?.debuggerHost ||  // Older Expo
    (Constants as any).manifest?.hostUri;  // Alternative

  if (hostUri) {
    // hostUri format is "IP:PORT" (e.g., "192.168.1.100:8081")
    const host = hostUri.split(':')[0];

    if (host && host !== 'localhost' && host !== '127.0.0.1') {
      const url = `http://${host}:${BACKEND_PORT}`;
      console.log('[API Config] Using Expo host:', url);
      return url;
    }
  }

  // Android emulator special case
  if (Platform.OS === 'android') {
    // 10.0.2.2 is the special IP for Android emulator to reach host machine
    const url = `http://10.0.2.2:${BACKEND_PORT}`;
    console.log('[API Config] Android emulator, using:', url);
    return url;
  }

  // iOS Simulator uses localhost
  if (Platform.OS === 'ios') {
    const url = `http://localhost:${BACKEND_PORT}`;
    console.log('[API Config] iOS simulator, using:', url);
    return url;
  }

  // Fallback
  console.warn('[API Config] Could not detect host, falling back to localhost');
  return `http://localhost:${BACKEND_PORT}`;
};

export const API_BASE_URL = getApiUrl();
export const API_URL = API_BASE_URL; // Alias for backwards compatibility

export const API_ENDPOINTS = {
  UPLOAD: `${API_BASE_URL}/upload/`,
  FULL_CLASSIFY: `${API_BASE_URL}/full_classify/`,
  CLASSIFY_BURN: `${API_BASE_URL}/classify-burn`,
  DERMOSCOPY_ANALYZE: `${API_BASE_URL}/dermoscopy/analyze`,
  ANALYSIS_HISTORY: `${API_BASE_URL}/analysis/history`,
  ANALYSIS_STATS: `${API_BASE_URL}/analysis/stats`,
  USER_PROFILE: `${API_BASE_URL}/profile`,
  USER_EXTENDED: `${API_BASE_URL}/me/extended`,
  // Async job queue endpoints
  JOBS_SUBMIT_FULL_CLASSIFY: `${API_BASE_URL}/jobs/submit/full-classify`,
  JOBS_SUBMIT_DERMOSCOPY: `${API_BASE_URL}/jobs/submit/dermoscopy`,
  JOBS_SUBMIT_BURN: `${API_BASE_URL}/jobs/submit/burn-classify`,
  JOBS_STATUS: `${API_BASE_URL}/jobs/status`,
  JOBS_RESULT: `${API_BASE_URL}/jobs/result`,
  JOBS_HEALTH: `${API_BASE_URL}/jobs/health`,
} as const;

export const REQUEST_TIMEOUT = 300000; // 300 seconds (5 minutes)

// Default export for screens using `import config from '../config'`
export default {
  API_URL: API_BASE_URL,
  apiUrl: API_BASE_URL,  // Lowercase alias for compatibility
  API_BASE_URL,
  API_ENDPOINTS,
  REQUEST_TIMEOUT,
};
