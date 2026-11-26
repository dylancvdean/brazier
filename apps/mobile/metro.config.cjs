const path = require('node:path');
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.watchFolders.push(path.resolve(__dirname, '../../packages/pairing'));
// The agent loop is the desktop's, run in-app; it has no dependencies of its own.
config.watchFolders.push(path.resolve(__dirname, '../desktop/src/agent'));

module.exports = config;
