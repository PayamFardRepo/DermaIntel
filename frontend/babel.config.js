module.exports = function(api) {
  api.cache(true);

  // Don't load reanimated plugin during Jest tests to avoid worklets dependency issue
  const isTest = process.env.NODE_ENV === 'test';

  return {
    presets: ['babel-preset-expo'],
    plugins: isTest ? [] : ['react-native-reanimated/plugin'],
  };
};
