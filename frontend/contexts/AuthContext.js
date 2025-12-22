import React, { createContext, useContext, useState, useEffect } from 'react';
import AuthService from '../services/AuthService';

console.log('🔧 [AuthContext] Module loaded');

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      setIsLoading(true);
      console.log('[AuthContext] Checking auth status...');
      const hasValidToken = await AuthService.loadStoredToken();
      console.log('[AuthContext] Token validation result:', hasValidToken);
      if (hasValidToken) {
        setUser(AuthService.getUser());
        console.log('[AuthContext] User loaded from stored token');
      } else {
        console.log('[AuthContext] No valid token found, user needs to login');
      }
    } catch (error) {
      console.error('[AuthContext] Auth check error:', error);
      // Don't throw - just let user proceed to login screen
    } finally {
      setIsLoading(false);
      console.log('[AuthContext] Auth check complete');
    }
  };

  const login = async (credentials) => {
    try {
      console.log('[AuthContext] Login started, setting isLoading=true');
      setIsLoading(true);
      console.log('[AuthContext] Calling AuthService.login...');
      const userData = await AuthService.login(credentials);
      console.log('[AuthContext] Login successful, user data:', userData);
      setUser(userData);
      console.log('[AuthContext] Setting isLoading=false');
      setIsLoading(false);
      console.log('[AuthContext] Login complete');
    } catch (error) {
      console.error('[AuthContext] Login error:', error);
      setIsLoading(false);
      throw error;
    }
  };

  const register = async (userData) => {
    try {
      setIsLoading(true);
      const newUser = await AuthService.register(userData);
      setUser(newUser);
      setIsLoading(false);
    } catch (error) {
      console.error('Registration error:', error);
      setIsLoading(false);
      throw error;
    }
  };

  const logout = async () => {
    try {
      await AuthService.logout();
      setUser(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const refreshUser = async () => {
    try {
      const userData = await AuthService.getCurrentUser();
      setUser(userData);
    } catch (error) {
      console.error('Refresh user error:', error);
      await logout();
    }
  };

  const value = {
    user,
    isAuthenticated: user !== null,
    isLoading,
    login,
    register,
    logout,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};