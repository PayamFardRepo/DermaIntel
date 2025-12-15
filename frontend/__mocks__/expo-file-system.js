/**
 * Mock for expo-file-system
 */

export const documentDirectory = 'file:///mock/documents/';
export const cacheDirectory = 'file:///mock/cache/';

export const getInfoAsync = jest.fn(async (fileUri) => {
  return {
    exists: true,
    isDirectory: false,
    size: 1024,
    modificationTime: Date.now(),
    uri: fileUri,
  };
});

export const readAsStringAsync = jest.fn(async (fileUri, options) => {
  return 'mock-file-content';
});

export const writeAsStringAsync = jest.fn(async (fileUri, contents, options) => {
  return Promise.resolve();
});

export const deleteAsync = jest.fn(async (fileUri, options) => {
  return Promise.resolve();
});

export const copyAsync = jest.fn(async ({ from, to }) => {
  return Promise.resolve();
});

export const moveAsync = jest.fn(async ({ from, to }) => {
  return Promise.resolve();
});

export const makeDirectoryAsync = jest.fn(async (fileUri, options) => {
  return Promise.resolve();
});

export const downloadAsync = jest.fn(async (uri, fileUri, options) => {
  return {
    uri: fileUri,
    status: 200,
    headers: {},
  };
});

export const uploadAsync = jest.fn(async (url, fileUri, options) => {
  return {
    status: 200,
    body: '{"success": true}',
    headers: {},
  };
});

export const EncodingType = {
  UTF8: 'utf8',
  Base64: 'base64',
};

export default {
  documentDirectory,
  cacheDirectory,
  getInfoAsync,
  readAsStringAsync,
  writeAsStringAsync,
  deleteAsync,
  copyAsync,
  moveAsync,
  makeDirectoryAsync,
  downloadAsync,
  uploadAsync,
  EncodingType,
};
