/**
 * Jest configuration for Skin Classifier Frontend
 *
 * Custom configuration for React Native/Expo compatibility
 */

module.exports = {
  // Use react-native preset instead of jest-expo to avoid setup.js issues
  preset: 'react-native',

  // Test file patterns
  testMatch: [
    '**/__tests__/**/*.test.[jt]s?(x)',
    '**/__tests__/**/*.spec.[jt]s?(x)',
  ],

  // Setup files
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],

  // Module name mapping for imports
  moduleNameMapper: {
    // Handle asset imports
    '\\.(jpg|jpeg|png|gif|webp|svg)$': '<rootDir>/__mocks__/fileMock.js',
    '\\.(css|less|scss|sass)$': '<rootDir>/__mocks__/styleMock.js',
    // Mock expo modules
    '^expo-secure-store$': '<rootDir>/__mocks__/expo-secure-store.js',
    '^expo-constants$': '<rootDir>/__mocks__/expo-constants.js',
    '^expo-image-picker$': '<rootDir>/__mocks__/expo-image-picker.js',
    '^expo-file-system$': '<rootDir>/__mocks__/expo-file-system.js',
    '^expo-router$': '<rootDir>/__mocks__/expo-router.js',
    '^@react-native-async-storage/async-storage$': '<rootDir>/__mocks__/@react-native-async-storage/async-storage.js',
    '^react-native-reanimated$': '<rootDir>/__mocks__/react-native-reanimated.js',
  },

  // Transform configuration
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|@unimodules/.*|unimodules|sentry-expo|native-base|react-native-svg|victory-native|i18next|react-i18next)',
  ],

  // Coverage configuration
  collectCoverageFrom: [
    'services/**/*.{js,ts}',
    'contexts/**/*.{js,ts,tsx}',
    'components/**/*.{js,ts,tsx}',
    'utils/**/*.{js,ts}',
    '!**/__tests__/**',
    '!**/node_modules/**',
  ],

  coverageThreshold: {
    global: {
      branches: 50,
      functions: 50,
      lines: 50,
      statements: 50,
    },
  },

  // Test environment
  testEnvironment: 'node',

  // Clear mocks between tests
  clearMocks: true,

  // Module directories
  moduleDirectories: ['node_modules', '<rootDir>'],

  // Verbose output
  verbose: true,

  // Timeout for async tests
  testTimeout: 10000,
};
