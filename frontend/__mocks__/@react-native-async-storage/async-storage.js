/**
 * Mock for @react-native-async-storage/async-storage
 */

const storage = new Map();

const AsyncStorage = {
  setItem: jest.fn(async (key, value) => {
    storage.set(key, value);
    return Promise.resolve();
  }),

  getItem: jest.fn(async (key) => {
    return Promise.resolve(storage.get(key) || null);
  }),

  removeItem: jest.fn(async (key) => {
    storage.delete(key);
    return Promise.resolve();
  }),

  clear: jest.fn(async () => {
    storage.clear();
    return Promise.resolve();
  }),

  getAllKeys: jest.fn(async () => {
    return Promise.resolve(Array.from(storage.keys()));
  }),

  multiGet: jest.fn(async (keys) => {
    return Promise.resolve(keys.map((key) => [key, storage.get(key) || null]));
  }),

  multiSet: jest.fn(async (keyValuePairs) => {
    keyValuePairs.forEach(([key, value]) => storage.set(key, value));
    return Promise.resolve();
  }),

  multiRemove: jest.fn(async (keys) => {
    keys.forEach((key) => storage.delete(key));
    return Promise.resolve();
  }),

  // Helper to clear storage between tests
  __clearStorage: () => {
    storage.clear();
  },
};

export default AsyncStorage;
