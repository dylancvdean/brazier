const fs = require('node:fs');
const path = require('node:path');
const { withAppDelegate } = require('expo/config-plugins');

// Expo SDK 54's template still starts its window in the app delegate. Keep the
// scene migration when prebuild regenerates that template, without upgrading
// the rest of the Expo/React Native stack as part of this launch fix.
module.exports = function withIosScenes(config) {
  return withAppDelegate(config, (config) => {
    if (config.modResults.language !== 'swift') {
      throw new Error('The iOS scene lifecycle plugin requires a Swift AppDelegate.');
    }
    let source = config.modResults.contents;
    if (!source.includes('self.launchOptions = launchOptions')) {
      const startup = /#if os\(iOS\) \|\| os\(tvOS\)\s+window = UIWindow\(frame: UIScreen\.main\.bounds\)[\s\S]*?#endif/;
      if (!startup.test(source) || !source.includes('var reactNativeFactory: RCTReactNativeFactory?')) {
        throw new Error('Expo AppDelegate template changed; review the iOS scene lifecycle migration.');
      }
      source = source.replace('var reactNativeFactory: RCTReactNativeFactory?',
        'var reactNativeFactory: RCTReactNativeFactory?\n  var launchOptions: [UIApplication.LaunchOptionsKey: Any]?');
      source = source.replace(startup, '    self.launchOptions = launchOptions');
    }
    const marker = '// UIKit creates the window only after connecting a scene.';
    const scene = fs.readFileSync(path.join(__dirname, 'SceneDelegate.swift'), 'utf8');
    source = source.split(marker)[0].trimEnd() + '\n\n' + scene.slice(scene.indexOf(marker));
    config.modResults.contents = source;
    return config;
  });
};
