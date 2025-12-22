import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL } from '../config';

console.log('🔧 [AuthService] Module loaded, API_BASE_URL:', API_BASE_URL);

// Helper function to test if backend is reachable
async function testBackendConnection() {
  try {
    console.log('[AuthService] Testing connection to:', `${API_BASE_URL}/health`);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    console.log('[AuthService] Backend health check:', response.status);
    return response.ok;
  } catch (error) {
    console.error('[AuthService] Backend not reachable:', error.message);
    return false;
  }
}

class AuthService {
  constructor() {
    this.token = null;
    this.user = null;
  }

  static getInstance() {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }

  async login(credentials) {
    try {
      console.log('[AuthService] Login starting...');
      console.log('[AuthService] Backend URL:', API_BASE_URL);

      // Test backend connectivity first
      const isReachable = await testBackendConnection();
      if (!isReachable) {
        throw new Error(`Cannot connect to backend at ${API_BASE_URL}. Please ensure:\n1. Backend server is running\n2. You're on the same network\n3. Firewall allows connections\n4. IP address in config.ts is correct`);
      }

      console.log('[AuthService] Backend is reachable, proceeding with login...');
      console.log('[AuthService] Connecting to:', `${API_BASE_URL}/login`);
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      console.log('[AuthService] Fetching login endpoint...');
      const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      console.log('[AuthService] Login response received:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Login failed');
      }

      const authData = await response.json();
      this.token = authData.access_token;
      console.log('[AuthService] Token received, storing...');

      // Store token securely
      await SecureStore.setItemAsync('auth_token', this.token);
      console.log('[AuthService] Token stored, getting user info...');

      // Get user info
      const user = await this.getCurrentUser();
      this.user = user;

      // Also cache user info for offline use
      await SecureStore.setItemAsync('cached_user', JSON.stringify(user));
      console.log('[AuthService] User info retrieved and cached:', user);

      // Return user object with token included
      return { ...user, token: this.token };
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error('[AuthService] Login timeout after 30 seconds');
        throw new Error('Login timeout - Cannot connect to server. Please check your network connection and ensure the backend is running.');
      }
      if (error.message && error.message.includes('Network request failed')) {
        console.error('[AuthService] Network error - cannot reach server');
        throw new Error('Network error - Cannot reach server at ' + API_BASE_URL + '. Please check that the backend is running and accessible.');
      }
      console.error('[AuthService] Login error:', error);
      throw error;
    }
  }

  async register(userData) {
    try {
      const response = await fetch(`${API_BASE_URL}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Registration failed');
      }

      const user = await response.json();

      // After successful registration, log them in
      // This will cache the token and user data
      await this.login({ username: userData.username, password: userData.password });

      return user;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  }

  async getCurrentUser() {
    if (!this.token) {
      throw new Error('No authentication token');
    }

    try {
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(`${API_BASE_URL}/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${this.token}`,
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        console.error('Get user info failed:', {
          status: response.status,
          statusText: response.statusText,
          error: errorData
        });
        throw new Error(errorData.detail || `Failed to get user info (${response.status})`);
      }

      const user = await response.json();
      this.user = user;
      return user;
    } catch (error) {
      // Handle timeout/abort errors
      if (error.name === 'AbortError') {
        throw new Error('Request timeout - server not responding');
      }
      // Don't log full error for authentication failures (expected when token expires)
      if (error.message && error.message.includes('Could not validate')) {
        throw new Error('Session expired');
      }
      console.error('Get user error:', error);
      throw error;
    }
  }

  async logout() {
    try {
      await SecureStore.deleteItemAsync('auth_token');
      await SecureStore.deleteItemAsync('cached_user');
      this.token = null;
      this.user = null;
    } catch (error) {
      console.error('Logout error:', error);
    }
  }

  async loadStoredToken() {
    try {
      console.log('[AuthService] Loading stored token...');
      const storedToken = await SecureStore.getItemAsync('auth_token');
      const cachedUser = await SecureStore.getItemAsync('cached_user');

      if (storedToken) {
        this.token = storedToken;

        // Load cached user first (for offline support)
        if (cachedUser) {
          try {
            this.user = JSON.parse(cachedUser);
            console.log('[AuthService] Loaded cached user:', this.user);
          } catch (e) {
            console.error('[AuthService] Failed to parse cached user:', e);
          }
        }

        console.log('[AuthService] Found stored token, validating with backend...');
        console.log('[AuthService] Backend URL:', API_BASE_URL);

        try {
          // Add timeout to prevent hanging if backend is unreachable
          const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Connection timeout')), 15000) // Increased to 15 seconds
          );
          const freshUser = await Promise.race([this.getCurrentUser(), timeoutPromise]);

          // Update cache with fresh user data
          this.user = freshUser;
          await SecureStore.setItemAsync('cached_user', JSON.stringify(freshUser));
          console.log('[AuthService] Token is valid, user refreshed successfully');
          return true;
        } catch (error) {
          console.log('[AuthService] Token validation failed:', error.message);

          // Only clear token if we got a definitive 401 Unauthorized or "Session expired" error
          // Don't clear on network timeouts - the token might still be valid
          if (error.message.includes('Session expired') ||
              error.message.includes('Could not validate') ||
              error.message.includes('401')) {
            console.log('[AuthService] Token is expired/invalid, clearing stored session');
            await SecureStore.deleteItemAsync('auth_token');
            await SecureStore.deleteItemAsync('cached_user');
            this.token = null;
            this.user = null;
            return false;
          } else {
            // Network error or timeout - keep the token and cached user data
            console.log('[AuthService] Network issue during validation, using cached user data');
            console.log('[AuthService] Will retry validation on next request');
            // We already have cached user loaded, so we can return true
            return this.user !== null; // Only return true if we have cached user
          }
        }
      }
      console.log('[AuthService] No stored token found');
      return false;
    } catch (error) {
      console.error('[AuthService] Load token error:', error);
      return false;
    }
  }

  getToken() {
    return this.token;
  }

  getUser() {
    if (this.user && this.token) {
      return { ...this.user, token: this.token };
    }
    return this.user;
  }

  isAuthenticated() {
    return this.token !== null && this.user !== null;
  }

  async makeAuthenticatedRequest(url, options = {}) {
    if (!this.token) {
      throw new Error('No authentication token');
    }

    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${this.token}`,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (response.status === 401) {
      // Token expired or invalid
      await this.logout();
      throw new Error('Authentication expired');
    }

    return response;
  }
}

export default AuthService.getInstance();