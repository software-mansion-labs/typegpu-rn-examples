/** @type {import('react-native-worklets/plugin').PluginOptions} */
const workletsPluginOptions = {
  bundleMode: true,
  strictGlobal: false, // optional, but recommended
  workletizableModules: ['typegpu'],
};

module.exports = (api) => {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      'unplugin-typegpu/babel',
      ['react-native-worklets/plugin', workletsPluginOptions],
    ],
  };
};
