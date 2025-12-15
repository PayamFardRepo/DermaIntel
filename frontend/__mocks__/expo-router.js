/**
 * Mock for expo-router
 */

export const useRouter = jest.fn(() => ({
  push: jest.fn(),
  replace: jest.fn(),
  back: jest.fn(),
  canGoBack: jest.fn(() => true),
  setParams: jest.fn(),
}));

export const useLocalSearchParams = jest.fn(() => ({}));

export const useSegments = jest.fn(() => []);

export const usePathname = jest.fn(() => '/');

export const Link = ({ children, href, ...props }) => children;

export const Redirect = jest.fn(() => null);

export const Stack = {
  Screen: jest.fn(({ children }) => children || null),
};

export const Tabs = {
  Screen: jest.fn(({ children }) => children || null),
};

export const router = {
  push: jest.fn(),
  replace: jest.fn(),
  back: jest.fn(),
  canGoBack: jest.fn(() => true),
  setParams: jest.fn(),
};

export default {
  useRouter,
  useLocalSearchParams,
  useSegments,
  usePathname,
  Link,
  Redirect,
  Stack,
  Tabs,
  router,
};
