/**
 * Mock for expo-secure-store
 *
 * Provides an in-memory storage implementation for testing
 */

const storage = new Map();

export const setItemAsync = jest.fn(async (key, value) => {
  storage.set(key, value);
  return Promise.resolve();
});

export const getItemAsync = jest.fn(async (key) => {
  return Promise.resolve(storage.get(key) || null);
});

export const deleteItemAsync = jest.fn(async (key) => {
  storage.delete(key);
  return Promise.resolve();
});

// Helper to clear storage between tests
export const __clearStorage = () => {
  storage.clear();
};

// Helper to get all stored items (for testing)
export const __getStorage = () => {
  return Object.fromEntries(storage);
};

export default {
  setItemAsync,
  getItemAsync,
  deleteItemAsync,
  __clearStorage,
  __getStorage,
};
