/**
 * Mock for react-native-reanimated
 */

const Reanimated = {
  useSharedValue: jest.fn((init) => ({ value: init })),
  useAnimatedStyle: jest.fn((fn) => fn()),
  useDerivedValue: jest.fn((fn) => ({ value: fn() })),
  useAnimatedGestureHandler: jest.fn((handlers) => handlers),
  withTiming: jest.fn((value) => value),
  withSpring: jest.fn((value) => value),
  withDecay: jest.fn((config) => 0),
  withDelay: jest.fn((delay, animation) => animation),
  withSequence: jest.fn((...animations) => animations[0]),
  withRepeat: jest.fn((animation) => animation),
  cancelAnimation: jest.fn(),
  runOnJS: jest.fn((fn) => fn),
  runOnUI: jest.fn((fn) => fn),
  createAnimatedComponent: (Component) => Component,
  Easing: {
    linear: jest.fn(),
    ease: jest.fn(),
    quad: jest.fn(),
    cubic: jest.fn(),
    poly: jest.fn(),
    sin: jest.fn(),
    circle: jest.fn(),
    exp: jest.fn(),
    elastic: jest.fn(),
    back: jest.fn(),
    bounce: jest.fn(),
    bezier: jest.fn(() => jest.fn()),
    in: jest.fn((easing) => easing),
    out: jest.fn((easing) => easing),
    inOut: jest.fn((easing) => easing),
  },
  FadeIn: { duration: jest.fn(() => ({ delay: jest.fn() })) },
  FadeOut: { duration: jest.fn(() => ({ delay: jest.fn() })) },
  SlideInRight: { duration: jest.fn(() => ({ delay: jest.fn() })) },
  SlideOutLeft: { duration: jest.fn(() => ({ delay: jest.fn() })) },
  Layout: { duration: jest.fn() },
  default: {
    View: 'Animated.View',
    Text: 'Animated.Text',
    Image: 'Animated.Image',
    ScrollView: 'Animated.ScrollView',
    FlatList: 'Animated.FlatList',
  },
  View: 'Animated.View',
  Text: 'Animated.Text',
  Image: 'Animated.Image',
  ScrollView: 'Animated.ScrollView',
  FlatList: 'Animated.FlatList',
};

module.exports = Reanimated;
module.exports.default = Reanimated;
