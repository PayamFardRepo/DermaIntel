import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../config';

const DISPLAY_MODE_KEY = '@display_mode';
const ACCOUNT_TYPE_KEY = '@account_type';
const AUTH_TOKEN_KEY = '@auth_token';

export type DisplayMode = 'simple' | 'professional';
export type AccountType = 'personal' | 'professional';

interface UserSettings {
  displayMode: DisplayMode;
  accountType: AccountType;
  isVerifiedProfessional: boolean;
  professionalLicenseNumber?: string;
  professionalLicenseState?: string;
  npiNumber?: string;
  verificationDate?: string;
}

interface UserSettingsContextType {
  settings: UserSettings;
  isLoading: boolean;
  setDisplayMode: (mode: DisplayMode) => Promise<void>;
  setAccountType: (type: AccountType) => Promise<void>;
  refreshSettings: () => Promise<void>;
  submitProfessionalVerification: (
    licenseNumber: string,
    licenseState: string,
    npiNumber?: string
  ) => Promise<{ success: boolean; message: string }>;
}

const defaultSettings: UserSettings = {
  displayMode: 'simple',
  accountType: 'personal',
  isVerifiedProfessional: false,
};

const UserSettingsContext = createContext<UserSettingsContextType | undefined>(undefined);

export const UserSettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<UserSettings>(defaultSettings);
  const [isLoading, setIsLoading] = useState(true);

  const getAuthToken = async (): Promise<string | null> => {
    try {
      return await AsyncStorage.getItem(AUTH_TOKEN_KEY);
    } catch {
      return null;
    }
  };

  // Load settings from server and local storage on mount
  const refreshSettings = useCallback(async () => {
    setIsLoading(true);
    try {
      // First try to get from local storage for immediate display
      const localDisplayMode = await AsyncStorage.getItem(DISPLAY_MODE_KEY);
      const localAccountType = await AsyncStorage.getItem(ACCOUNT_TYPE_KEY);

      if (localDisplayMode || localAccountType) {
        setSettings(prev => ({
          ...prev,
          displayMode: (localDisplayMode as DisplayMode) || prev.displayMode,
          accountType: (localAccountType as AccountType) || prev.accountType,
        }));
      }

      // Then try to sync with server
      const token = await getAuthToken();
      if (token) {
        const response = await fetch(`${API_URL}/me/professional-status`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          const newSettings: UserSettings = {
            displayMode: data.display_mode || 'simple',
            accountType: data.account_type || 'personal',
            isVerifiedProfessional: data.is_verified_professional || false,
            professionalLicenseNumber: data.professional_license_number,
            professionalLicenseState: data.professional_license_state,
            npiNumber: data.npi_number,
            verificationDate: data.verification_date,
          };

          setSettings(newSettings);

          // Cache locally
          await AsyncStorage.setItem(DISPLAY_MODE_KEY, newSettings.displayMode);
          await AsyncStorage.setItem(ACCOUNT_TYPE_KEY, newSettings.accountType);
        }
      }
    } catch (error) {
      console.error('Error loading user settings:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshSettings();
  }, [refreshSettings]);

  const setDisplayMode = async (mode: DisplayMode) => {
    try {
      // Update local state immediately
      setSettings(prev => ({ ...prev, displayMode: mode }));
      await AsyncStorage.setItem(DISPLAY_MODE_KEY, mode);

      // Sync with server
      const token = await getAuthToken();
      if (token) {
        const response = await fetch(`${API_URL}/me/settings?display_mode=${mode}`, {
          method: 'PUT',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          console.error('Failed to sync display mode with server');
        }
      }
    } catch (error) {
      console.error('Error setting display mode:', error);
    }
  };

  const setAccountType = async (type: AccountType) => {
    try {
      // Update local state immediately
      setSettings(prev => ({ ...prev, accountType: type }));
      await AsyncStorage.setItem(ACCOUNT_TYPE_KEY, type);

      // Sync with server
      const token = await getAuthToken();
      if (token) {
        const response = await fetch(`${API_URL}/me/settings?account_type=${type}`, {
          method: 'PUT',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          console.error('Failed to sync account type with server');
        }
      }
    } catch (error) {
      console.error('Error setting account type:', error);
    }
  };

  const submitProfessionalVerification = async (
    licenseNumber: string,
    licenseState: string,
    npiNumber?: string
  ): Promise<{ success: boolean; message: string }> => {
    try {
      const token = await getAuthToken();
      if (!token) {
        return { success: false, message: 'Not authenticated' };
      }

      const formData = new FormData();
      formData.append('license_number', licenseNumber);
      formData.append('license_state', licenseState);
      if (npiNumber) {
        formData.append('npi_number', npiNumber);
      }

      const response = await fetch(`${API_URL}/me/professional-verification`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        // Refresh settings to get updated status
        await refreshSettings();
        return { success: true, message: data.message || 'Verification request submitted' };
      } else {
        return { success: false, message: data.detail || 'Verification failed' };
      }
    } catch (error) {
      console.error('Error submitting professional verification:', error);
      return { success: false, message: 'Network error' };
    }
  };

  return (
    <UserSettingsContext.Provider
      value={{
        settings,
        isLoading,
        setDisplayMode,
        setAccountType,
        refreshSettings,
        submitProfessionalVerification,
      }}
    >
      {children}
    </UserSettingsContext.Provider>
  );
};

export const useUserSettings = () => {
  const context = useContext(UserSettingsContext);
  if (context === undefined) {
    throw new Error('useUserSettings must be used within a UserSettingsProvider');
  }
  return context;
};
