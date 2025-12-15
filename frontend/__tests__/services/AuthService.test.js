/**
 * Tests for AuthService
 *
 * Tests authentication functionality including:
 * - Login/logout
 * - Token management
 * - User data handling
 * - Network error handling
 */

import * as SecureStore from 'expo-secure-store';

// Mock config before importing AuthService
jest.mock('../../config', () => ({
  API_BASE_URL: 'http://test-api.example.com',
}));

// Import after mocks are set up
import AuthService from '../../services/AuthService';

describe('AuthService', () => {
  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    SecureStore.__clearStorage();

    // Reset AuthService state
    AuthService.token = null;
    AuthService.user = null;
  });

  describe('login', () => {
    it('should successfully login with valid credentials', async () => {
      // Mock health check
      global.fetch
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
        })
        // Mock login response
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            access_token: 'test-token-123',
            token_type: 'bearer',
          }),
        })
        // Mock /me response
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            id: 1,
            username: 'testuser',
            email: 'test@example.com',
            is_active: true,
          }),
        });

      const result = await AuthService.login({
        username: 'testuser',
        password: 'password123',
      });

      expect(result).toHaveProperty('username', 'testuser');
      expect(result).toHaveProperty('token', 'test-token-123');
      expect(AuthService.getToken()).toBe('test-token-123');
      expect(SecureStore.setItemAsync).toHaveBeenCalledWith('auth_token', 'test-token-123');
    });

    it('should throw error for invalid credentials', async () => {
      // Mock health check
      global.fetch
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
        })
        // Mock failed login response
        .mockResolvedValueOnce({
          ok: false,
          status: 401,
          json: () => Promise.resolve({
            detail: 'Incorrect username or password',
          }),
        });

      await expect(
        AuthService.login({
          username: 'wronguser',
          password: 'wrongpassword',
        })
      ).rejects.toThrow('Incorrect username or password');
    });

    it('should throw error when backend is unreachable', async () => {
      // Mock health check failure
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      await expect(
        AuthService.login({
          username: 'testuser',
          password: 'password123',
        })
      ).rejects.toThrow(/Cannot connect to backend/);
    });

    it('should handle network timeout', async () => {
      // Mock health check
      global.fetch
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
        })
        // Mock timeout by rejecting with AbortError
        .mockRejectedValueOnce(Object.assign(new Error('Aborted'), { name: 'AbortError' }));

      await expect(
        AuthService.login({
          username: 'testuser',
          password: 'password123',
        })
      ).rejects.toThrow(/timeout/i);
    });
  });

  describe('register', () => {
    it('should successfully register a new user', async () => {
      // Mock registration
      global.fetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            id: 2,
            username: 'newuser',
            email: 'new@example.com',
          }),
        })
        // Mock health check for login
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
        })
        // Mock login after registration
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            access_token: 'new-token-456',
            token_type: 'bearer',
          }),
        })
        // Mock /me response
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            id: 2,
            username: 'newuser',
            email: 'new@example.com',
            is_active: true,
          }),
        });

      const result = await AuthService.register({
        username: 'newuser',
        email: 'new@example.com',
        password: 'newpassword123',
      });

      expect(result).toHaveProperty('username', 'newuser');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/register'),
        expect.any(Object)
      );
    });

    it('should throw error for duplicate username', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          detail: 'Username already registered',
        }),
      });

      await expect(
        AuthService.register({
          username: 'existinguser',
          email: 'new@example.com',
          password: 'password123',
        })
      ).rejects.toThrow('Username already registered');
    });
  });

  describe('logout', () => {
    it('should clear token and user data', async () => {
      // Set up initial state
      AuthService.token = 'test-token';
      AuthService.user = { id: 1, username: 'testuser' };
      await SecureStore.setItemAsync('auth_token', 'test-token');
      await SecureStore.setItemAsync('cached_user', JSON.stringify({ id: 1 }));

      await AuthService.logout();

      expect(AuthService.getToken()).toBeNull();
      expect(AuthService.getUser()).toBeNull();
      expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith('auth_token');
      expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith('cached_user');
    });
  });

  describe('getCurrentUser', () => {
    it('should fetch user data with valid token', async () => {
      AuthService.token = 'valid-token';

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 1,
          username: 'testuser',
          email: 'test@example.com',
          is_active: true,
        }),
      });

      const user = await AuthService.getCurrentUser();

      expect(user).toHaveProperty('username', 'testuser');
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/me'),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer valid-token',
          }),
        })
      );
    });

    it('should throw error when no token is set', async () => {
      AuthService.token = null;

      await expect(AuthService.getCurrentUser()).rejects.toThrow('No authentication token');
    });

    it('should handle session expired error', async () => {
      AuthService.token = 'expired-token';

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          detail: 'Could not validate credentials',
        }),
      });

      await expect(AuthService.getCurrentUser()).rejects.toThrow('Session expired');
    });
  });

  describe('loadStoredToken', () => {
    it('should load and validate stored token', async () => {
      // Set up stored token
      await SecureStore.setItemAsync('auth_token', 'stored-token');
      await SecureStore.setItemAsync('cached_user', JSON.stringify({
        id: 1,
        username: 'cacheduser',
      }));

      // Mock /me response
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 1,
          username: 'cacheduser',
          email: 'cached@example.com',
          is_active: true,
        }),
      });

      const result = await AuthService.loadStoredToken();

      expect(result).toBe(true);
      expect(AuthService.getToken()).toBe('stored-token');
    });

    it('should return false when no token is stored', async () => {
      const result = await AuthService.loadStoredToken();

      expect(result).toBe(false);
      expect(AuthService.getToken()).toBeNull();
    });

    it('should clear invalid token and return false', async () => {
      await SecureStore.setItemAsync('auth_token', 'invalid-token');

      // Mock failed validation with session expired error
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          detail: 'Could not validate credentials',
        }),
      });

      const result = await AuthService.loadStoredToken();

      expect(result).toBe(false);
      expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith('auth_token');
    });

    it('should use cached user data on network timeout', async () => {
      await SecureStore.setItemAsync('auth_token', 'stored-token');
      await SecureStore.setItemAsync('cached_user', JSON.stringify({
        id: 1,
        username: 'cacheduser',
      }));

      // Mock timeout
      global.fetch.mockRejectedValueOnce(new Error('Connection timeout'));

      const result = await AuthService.loadStoredToken();

      // Should return true because cached user exists
      expect(result).toBe(true);
    });
  });

  describe('isAuthenticated', () => {
    it('should return true when token and user are set', () => {
      AuthService.token = 'test-token';
      AuthService.user = { id: 1 };

      expect(AuthService.isAuthenticated()).toBe(true);
    });

    it('should return false when token is missing', () => {
      AuthService.token = null;
      AuthService.user = { id: 1 };

      expect(AuthService.isAuthenticated()).toBe(false);
    });

    it('should return false when user is missing', () => {
      AuthService.token = 'test-token';
      AuthService.user = null;

      expect(AuthService.isAuthenticated()).toBe(false);
    });
  });

  describe('makeAuthenticatedRequest', () => {
    it('should add authorization header to requests', async () => {
      AuthService.token = 'test-token';

      global.fetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ data: 'test' }),
      });

      await AuthService.makeAuthenticatedRequest('http://test-api.example.com/endpoint', {
        method: 'GET',
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://test-api.example.com/endpoint',
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      );
    });

    it('should throw error when not authenticated', async () => {
      AuthService.token = null;

      await expect(
        AuthService.makeAuthenticatedRequest('http://test-api.example.com/endpoint')
      ).rejects.toThrow('No authentication token');
    });

    it('should logout on 401 response', async () => {
      AuthService.token = 'expired-token';
      AuthService.user = { id: 1 };

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
      });

      await expect(
        AuthService.makeAuthenticatedRequest('http://test-api.example.com/endpoint')
      ).rejects.toThrow('Authentication expired');

      expect(AuthService.getToken()).toBeNull();
    });
  });

  describe('getUser', () => {
    it('should return user with token when both exist', () => {
      AuthService.token = 'test-token';
      AuthService.user = { id: 1, username: 'testuser' };

      const user = AuthService.getUser();

      expect(user).toHaveProperty('id', 1);
      expect(user).toHaveProperty('username', 'testuser');
      expect(user).toHaveProperty('token', 'test-token');
    });

    it('should return just user when no token', () => {
      AuthService.token = null;
      AuthService.user = { id: 1, username: 'testuser' };

      const user = AuthService.getUser();

      expect(user).toHaveProperty('id', 1);
      expect(user).not.toHaveProperty('token');
    });

    it('should return null when no user', () => {
      AuthService.token = 'test-token';
      AuthService.user = null;

      expect(AuthService.getUser()).toBeNull();
    });
  });
});
