/**
 * Mock for expo-image-picker
 */

export const MediaTypeOptions = {
  All: 'All',
  Videos: 'Videos',
  Images: 'Images',
};

export const CameraType = {
  front: 'front',
  back: 'back',
};

export const launchImageLibraryAsync = jest.fn(async (options) => {
  return {
    canceled: false,
    assets: [
      {
        uri: 'file:///mock/image.jpg',
        width: 1920,
        height: 1080,
        type: 'image',
        fileName: 'mock-image.jpg',
        fileSize: 1024000,
        base64: 'mock-base64-data',
      },
    ],
  };
});

export const launchCameraAsync = jest.fn(async (options) => {
  return {
    canceled: false,
    assets: [
      {
        uri: 'file:///mock/camera-photo.jpg',
        width: 1920,
        height: 1080,
        type: 'image',
        fileName: 'camera-photo.jpg',
        fileSize: 2048000,
        base64: 'mock-camera-base64-data',
      },
    ],
  };
});

export const requestMediaLibraryPermissionsAsync = jest.fn(async () => {
  return { status: 'granted', granted: true };
});

export const requestCameraPermissionsAsync = jest.fn(async () => {
  return { status: 'granted', granted: true };
});

export default {
  MediaTypeOptions,
  CameraType,
  launchImageLibraryAsync,
  launchCameraAsync,
  requestMediaLibraryPermissionsAsync,
  requestCameraPermissionsAsync,
};
