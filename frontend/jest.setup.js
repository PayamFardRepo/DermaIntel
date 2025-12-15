/**
 * Jest setup file for Skin Classifier Frontend
 *
 * Configures global mocks and test utilities
 */

// Suppress console logs during tests (optional - comment out for debugging)
// global.console = {
//   ...console,
//   log: jest.fn(),
//   debug: jest.fn(),
//   info: jest.fn(),
//   warn: jest.fn(),
// };

// Mock fetch globally
global.fetch = jest.fn();

// Mock AbortController
global.AbortController = class AbortController {
  constructor() {
    this.signal = {
      aborted: false,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    };
  }
  abort() {
    this.signal.aborted = true;
  }
};

// Reset mocks before each test
beforeEach(() => {
  jest.clearAllMocks();
  global.fetch.mockClear();
});

// Clean up after all tests
afterAll(() => {
  jest.restoreAllMocks();
});

// Add custom matchers
expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// Mock __DEV__ for React Native
global.__DEV__ = true;
